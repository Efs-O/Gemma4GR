"""
Step 4 — Build JSONL dataset for Gemma 4 audio fine-tuning.
Input:  data/resampled_audio/*.wav  (16kHz, synthetic Q/A pairs)
        data/transcripts/*.txt
        data/human_voice_dataset/wavs/*.wav  (22050Hz, resampled on the fly)
        data/human_voice_dataset/metadata.csv  (stem|transcript)
Output: data/train_stt.jsonl
        data/train_stt_val.jsonl
"""
import json, os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import soundfile as sf
import librosa

load_dotenv()

BASE           = Path(__file__).parent.parent
STT_DIR        = BASE / "data" / "resampled_audio"
TXT_DIR        = BASE / "data" / "transcripts"
HUMAN_WAV_DIR  = BASE / "data" / "human_voice_dataset" / "wavs"
HUMAN_META     = BASE / "data" / "human_voice_dataset" / "metadata.csv"
RESAMPLED_DIR  = BASE / "data" / "human_voice_resampled"
OUT_TRAIN      = BASE / "data" / "train_stt.jsonl"
OUT_VAL        = BASE / "data" / "train_stt_val.jsonl"
VAL_RATIO      = 0.05
STT_SAMPLE_RATE = 16000

SYSTEM_PROMPT = "Transcribe the following speech segment in Greek into Greek text."


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


def _make_entry(wav: Path, transcript: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(wav.resolve())},
                    {"type": "text",  "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": transcript}],
            },
        ]
    }


def load_synthetic_pairs() -> list[dict]:
    pairs = []
    for wav in sorted(STT_DIR.glob("*.wav")):
        txt = TXT_DIR / (wav.stem + ".txt")
        if not txt.exists():
            continue
        transcript = txt.read_text(encoding="utf-8").strip()
        if transcript:
            pairs.append(_make_entry(wav, transcript))
    return pairs


def load_human_pairs() -> list[dict]:
    if not HUMAN_META.exists() or not HUMAN_WAV_DIR.exists():
        print(f"  [SKIP] Human voice dataset not found, skipping.")
        return []
    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    pairs = []
    with open(HUMAN_META, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            stem, transcript = line.split("|", 1)
            transcript = transcript.strip()
            if not transcript:
                continue
            src_wav = HUMAN_WAV_DIR / f"{stem}.wav"
            if not src_wav.exists():
                continue
            dst_wav = RESAMPLED_DIR / f"{stem}.wav"
            _resample_and_save(src_wav, dst_wav)
            pairs.append(_make_entry(dst_wav, transcript))
    return pairs


def build():
    synthetic = load_synthetic_pairs()
    human     = load_human_pairs()
    pairs     = human

    if not pairs:
        print("[ERROR] No audio/transcript pairs found.")
        return

    print(f"  Synthetic Q/A pairs : {len(synthetic)}")
    print(f"  Human sentence pairs: {len(human)}")
    print(f"  Total               : {len(pairs)}")

    split     = max(1, int(len(pairs) * (1 - VAL_RATIO)))
    train_set = pairs[:split]
    val_set   = pairs[split:]

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for entry in train_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for entry in val_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ STT dataset ready.")
    print(f"  Train: {len(train_set)} pairs → {OUT_TRAIN}")
    print(f"  Val:   {len(val_set)} pairs  → {OUT_VAL}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Gemma4GR — Prepare STT Dataset")
    print("=" * 55 + "\n")
    build()
