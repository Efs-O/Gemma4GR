"""
Track C Step 2 — Build Gemma audio JSONL from synthesized Q&A WAVs.

Input:  data/qa_audio/manifest.jsonl   (from synthesize_qa_audio.py)
Output: data/train_stt_qa.jsonl
        data/val_stt_qa.jsonl

Format: Gemma 4 multimodal audio chat — audio of question → text answer.

Run:
  python training/prepare_stt_qa_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MANIFEST = DATA_DIR / "qa_audio" / "manifest.jsonl"
TRAIN_OUT = DATA_DIR / "train_stt_qa.jsonl"
VAL_OUT = DATA_DIR / "val_stt_qa.jsonl"

VAL_RATIO = 0.10
SYSTEM_PROMPT = "Άκουσε την ερώτηση και δώσε μια ολοκληρωμένη απάντηση στα Ελληνικά."


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST}")
        print("  Run synthesize_qa_audio.py first.")
        sys.exit(1)

    entries = []
    skipped = 0
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            wav_path = Path(obj.get("wav_path", ""))
            answer = (obj.get("answer") or "").strip()
            if not wav_path.exists() or not answer:
                skipped += 1
                continue

            entries.append({
                "wav_path": str(wav_path),
                "question": obj.get("question", ""),
                "answer": answer,
                "category": obj.get("category", "general"),
            })

    print(f"  Manifest entries: {len(entries)} valid | {skipped} skipped")
    return entries


def make_record(entry: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": entry["wav_path"]},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": entry["answer"],
            },
        ]
    }


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(make_record(entry), ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 60)
    print("  Gemma4GR Track C — Prepare STT Q&A Dataset")
    print("=" * 60 + "\n")

    entries = load_manifest()

    random.seed(42)
    shuffled = entries.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = shuffled[:n_val]
    train_set = shuffled[n_val:]

    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    write_jsonl(TRAIN_OUT, train_set)
    write_jsonl(VAL_OUT, val_set)

    cats: dict[str, int] = {}
    for e in entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1

    print("\n  Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: (-x[1], x[0])):
        bar = "#" * (count // 10)
        print(f"    {cat:<25} {count:4d}  {bar}")

    print("\n  Saved:")
    print(f"    {TRAIN_OUT}")
    print(f"    {VAL_OUT}")
    print("\n  Next: python training/train_stt_qa_local.py")


if __name__ == "__main__":
    main()
