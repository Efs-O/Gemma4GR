"""
Build a strict Greek STT-only dataset from the human voice corpus.

This does not replace the older datasets. It creates a new final-run dataset:
  - data/train_stt_final.jsonl
  - data/val_stt_final.jsonl
  - data/train_stt_final_report.json
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import librosa
import soundfile as sf
from dotenv import load_dotenv

from console_encoding import ensure_utf8_console

load_dotenv()
ensure_utf8_console()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
HUMAN_WAV_DIR = DATA_DIR / "human_voice_dataset" / "wavs"
HUMAN_META = DATA_DIR / "human_voice_dataset" / "metadata.csv"
RESAMPLED_DIR = DATA_DIR / "human_voice_resampled"
TRAIN_OUT = DATA_DIR / "train_stt_final.jsonl"
VAL_OUT = DATA_DIR / "val_stt_final.jsonl"
REPORT_OUT = DATA_DIR / "train_stt_final_report.json"
PROMPT_PATH = BASE / "greek_stt_prompt.txt"
STT_SAMPLE_RATE = 16000
VAL_RATIO = 0.10
SHUFFLE_SEED = 42

FILLER_PATTERNS = [
    r"\bείμαι έτοιμ[οςη]\b",
    r"\bμπορ[ωώ] να ακούσω\b",
    r"\bάκουσε\b",
    r"\bαπάντησε\b",
    r"\bμεταγραφή\b",
    r"\btranscribe\b",
    r"\btranscription\b",
]


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"STT prompt file is empty: {PROMPT_PATH}")
    return text


def backup_file(path: Path, tag: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.{tag}{path.suffix}")
    path.replace(backup)
    return backup


def greek_char_ratio(text: str) -> float:
    chars = [ch for ch in text if ch.isalpha()]
    if not chars:
        return 0.0
    greek = [
        ch
        for ch in chars
        if "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff"
    ]
    return len(greek) / len(chars)


def normalize_transcript(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_transcript(text: str, prompt_text: str) -> tuple[bool, str | None, str]:
    normalized = normalize_transcript(text)
    if not normalized:
        return False, "empty", normalized
    if normalized == prompt_text:
        return False, "echoed_prompt", normalized
    if len(normalized) < 2:
        return False, "too_short", normalized
    if len(normalized) > 280:
        return False, "too_long", normalized
    if greek_char_ratio(normalized) < 0.55:
        return False, "low_greek_ratio", normalized
    lowered = normalized.lower()
    for pattern in FILLER_PATTERNS:
        if re.search(pattern, lowered):
            return False, "assistant_filler", normalized
    return True, None, normalized


def _resample_and_save(src: Path, dst: Path) -> Path:
    if dst.exists():
        return dst
    data, sr = sf.read(str(src), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != STT_SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=STT_SAMPLE_RATE)
    sf.write(str(dst), data, STT_SAMPLE_RATE)
    return dst


def make_record(wav_path: Path, prompt_text: str, transcript: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(wav_path.resolve())},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": transcript}],
            },
        ]
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    prompt_text = load_prompt()
    if not HUMAN_META.exists():
        raise FileNotFoundError(f"Missing metadata: {HUMAN_META}")
    if not HUMAN_WAV_DIR.exists():
        raise FileNotFoundError(f"Missing WAV folder: {HUMAN_WAV_DIR}")

    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    rejected = Counter()
    missing_wavs = 0

    with HUMAN_META.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "|" not in line:
                continue
            stem, transcript = line.split("|", 1)
            src_wav = HUMAN_WAV_DIR / f"{stem}.wav"
            if not src_wav.exists():
                missing_wavs += 1
                continue
            ok, reason, cleaned = validate_transcript(transcript, prompt_text)
            if not ok:
                rejected[reason or "unknown"] += 1
                continue
            dst_wav = RESAMPLED_DIR / f"{stem}.wav"
            _resample_and_save(src_wav, dst_wav)
            kept.append({"stem": stem, "wav_path": dst_wav, "transcript": cleaned})

    if not kept:
        raise RuntimeError("No valid STT-only rows were produced.")

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * VAL_RATIO))
    val_rows = kept[:n_val]
    train_rows = kept[n_val:]

    train_records = [make_record(item["wav_path"], prompt_text, item["transcript"]) for item in train_rows]
    val_records = [make_record(item["wav_path"], prompt_text, item["transcript"]) for item in val_rows]

    train_backup = backup_file(TRAIN_OUT, "pre_final_rebuild")
    val_backup = backup_file(VAL_OUT, "pre_final_rebuild")
    write_jsonl(TRAIN_OUT, train_records)
    write_jsonl(VAL_OUT, val_records)

    report = {
        "prompt_path": str(PROMPT_PATH),
        "source_metadata": str(HUMAN_META),
        "source_wavs": str(HUMAN_WAV_DIR),
        "train_out": str(TRAIN_OUT),
        "val_out": str(VAL_OUT),
        "kept_total": len(kept),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "missing_wavs": missing_wavs,
        "rejected": dict(rejected),
        "backups": {
            "train": str(train_backup) if train_backup else None,
            "val": str(val_backup) if val_backup else None,
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("Gemma4GR - Prepare Final STT Dataset")
    print("=" * 60)
    print(f"Prompt       : {PROMPT_PATH}")
    print(f"Kept         : {len(kept)}")
    print(f"Train        : {len(train_records)} -> {TRAIN_OUT}")
    print(f"Val          : {len(val_records)} -> {VAL_OUT}")
    print(f"Rejected     : {dict(rejected)}")
    print(f"Missing WAVs : {missing_wavs}")
    print(f"Report       : {REPORT_OUT}")


if __name__ == "__main__":
    main()
