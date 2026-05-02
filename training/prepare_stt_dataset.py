"""
Step 4 — Build JSONL dataset for Gemma 4 audio fine-tuning.
Input:  data/resampled_audio/*.wav  (16kHz)
        data/transcripts/*.txt
Output: data/train_stt.jsonl
        data/train_stt_val.jsonl
"""
import json, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE      = Path(__file__).parent.parent
STT_DIR   = BASE / "data" / "resampled_audio"
TXT_DIR   = BASE / "data" / "transcripts"
OUT_TRAIN = BASE / "data" / "train_stt.jsonl"
OUT_VAL   = BASE / "data" / "train_stt_val.jsonl"
VAL_RATIO = 0.05   # 5% validation

SYSTEM_PROMPT = "Transcribe the following speech segment in Greek into Greek text."


def build():
    pairs = []
    for wav in sorted(STT_DIR.glob("*.wav")):
        txt = TXT_DIR / (wav.stem + ".txt")
        if not txt.exists():
            continue
        transcript = txt.read_text(encoding="utf-8").strip()
        if not transcript:
            continue

        entry = {
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
                    "content": transcript,
                },
            ]
        }
        pairs.append(entry)

    if not pairs:
        print("[ERROR] No audio/transcript pairs found.")
        print(f"  Expected WAVs in: {STT_DIR}")
        print(f"  Expected TXTs in: {TXT_DIR}")
        return

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
