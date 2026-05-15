"""
Track C Step 2 - Build Gemma audio JSONL from two sources:
  1. Human voice recordings  (transcription task)
  2. JOY-synthesized QA audio pairs  (comprehension task)

Inputs:
  data/human_voice_dataset/wavs/*.wav  +  metadata.csv
  data/qa_audio/manifest.jsonl          (set STT_QA_SKIP_QA_AUDIO=1 to omit)
Output:
  data/train_stt_qa.jsonl
  data/val_stt_qa.jsonl

Run:
  python training/prepare_stt_qa_dataset.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf
from dotenv import load_dotenv

from console_encoding import ensure_utf8_console

load_dotenv()
ensure_utf8_console()

BASE             = Path(__file__).parent.parent
DATA_DIR         = BASE / "data"
HUMAN_WAV_DIR    = DATA_DIR / "human_voice_dataset" / "wavs"
HUMAN_META       = DATA_DIR / "human_voice_dataset" / "metadata.csv"
RESAMPLED_DIR    = DATA_DIR / "human_voice_resampled"
STT_SAMPLE_RATE  = 16000
QA_AUDIO_MANIFEST = DATA_DIR / "qa_audio" / "manifest.jsonl"
SKIP_QA_AUDIO = os.getenv("STT_QA_SKIP_QA_AUDIO", "0").strip().lower() in ("1", "true", "yes")

STT_INSTRUCTION = (
    "Άκουσε προσεκτικά την ομιλία και γράψε ακριβώς στα Ελληνικά αυτό που ακούστηκε. "
    "Μην προσθέσεις επεξηγήσεις."
)
QA_INSTRUCTION = "Άκουσε την ερώτηση και δώσε μια ολοκληρωμένη απάντηση στα Ελληνικά."

_train_out_raw = os.getenv("STT_QA_TRAIN_OUT", "").strip()
TRAIN_OUT = Path(_train_out_raw) if _train_out_raw else DATA_DIR / "train_stt_qa.jsonl"
if not TRAIN_OUT.is_absolute():
    TRAIN_OUT = BASE / TRAIN_OUT
_val_out_raw = os.getenv("STT_QA_VAL_OUT", "").strip()
VAL_OUT = Path(_val_out_raw) if _val_out_raw else DATA_DIR / "val_stt_qa.jsonl"
if not VAL_OUT.is_absolute():
    VAL_OUT = BASE / VAL_OUT

VAL_RATIO = 0.10


def backup_file(path: Path, tag: str) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.{tag}.{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


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


def load_human_entries() -> list[dict]:
    if not HUMAN_META.exists():
        print(f"[ERROR] metadata.csv not found: {HUMAN_META}")
        sys.exit(1)
    if not HUMAN_WAV_DIR.exists():
        print(f"[ERROR] WAV folder not found: {HUMAN_WAV_DIR}")
        sys.exit(1)

    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    skipped = 0

    with open(HUMAN_META, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            stem, transcript = line.split("|", 1)
            transcript = transcript.strip()
            if not transcript:
                skipped += 1
                continue
            src_wav = HUMAN_WAV_DIR / f"{stem}.wav"
            if not src_wav.exists():
                skipped += 1
                continue
            dst_wav = RESAMPLED_DIR / f"{stem}.wav"
            _resample_and_save(src_wav, dst_wav)
            entries.append({"wav_path": str(dst_wav), "transcript": transcript})

    print(f"  Human entries: {len(entries)} loaded | {skipped} skipped")
    return entries


def load_qa_audio_entries() -> list[dict]:
    if not QA_AUDIO_MANIFEST.exists():
        print(f"  [WARN] QA audio manifest not found, skipping: {QA_AUDIO_MANIFEST}")
        return []
    entries = []
    skipped = 0
    with open(QA_AUDIO_MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            wav = Path(item["wav_path"])
            if not wav.exists():
                skipped += 1
                continue
            answer = item.get("answer", "").strip()
            if not answer:
                skipped += 1
                continue
            entries.append({"wav_path": str(wav), "answer": answer, "task": "qa"})
    print(f"  QA audio entries: {len(entries)} loaded | {skipped} skipped")
    return entries


def make_record(entry: dict) -> dict:
    if entry.get("task") == "qa":
        instruction = QA_INSTRUCTION
        answer_text = entry["answer"]
    else:
        instruction = STT_INSTRUCTION
        answer_text = entry["transcript"]
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": entry["wav_path"]},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]
    }


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(make_record(entry), ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 60)
    print("  Gemma4GR Track C - Prepare Combined STT Dataset")
    print("=" * 60 + "\n")

    human_entries = load_human_entries()

    qa_entries: list[dict] = []
    if not SKIP_QA_AUDIO:
        qa_entries = load_qa_audio_entries()
    else:
        print("  QA audio skipped (STT_QA_SKIP_QA_AUDIO=1)")

    all_entries = human_entries + qa_entries
    print(f"\n  Total combined: {len(all_entries)} "
          f"(human={len(human_entries)}, qa_audio={len(qa_entries)})")

    random.seed(42)
    shuffled = all_entries.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = shuffled[:n_val]
    train_set = shuffled[n_val:]

    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    train_backup = backup_file(TRAIN_OUT, "pre_rebuild")
    val_backup = backup_file(VAL_OUT, "pre_rebuild")
    write_jsonl(TRAIN_OUT, train_set)
    write_jsonl(VAL_OUT, val_set)

    print("\n  Saved:")
    print(f"    {TRAIN_OUT}")
    print(f"    {VAL_OUT}")
    if train_backup:
        print(f"    train backup: {train_backup}")
    if val_backup:
        print(f"    val backup:   {val_backup}")
    print("\n  Next: python training/train_stt_qa_local.py")


if __name__ == "__main__":
    main()
