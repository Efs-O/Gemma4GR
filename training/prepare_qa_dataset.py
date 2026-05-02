"""
Phase 2 - Prepare Q&A pairs into chat-format JSONL for text LoRA training.

Input:  data/qa_pairs.jsonl by default, or QA_DATASET_SOURCE
Output: data/train_qa.jsonl (90% split, chat format)
        data/val_qa.jsonl   (10% split)

Format per line (Gemma chat template):
  {"text": "<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn><eos>"}
"""
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
IN_FILE = DATA_DIR / os.getenv("QA_DATASET_SOURCE", "qa_pairs.jsonl")
TRAIN_OUT = DATA_DIR / "train_qa.jsonl"
VAL_OUT = DATA_DIR / "val_qa.jsonl"

VAL_RATIO = 0.10

BOS = "<bos>"
TURN_START = "<start_of_turn>"
TURN_END = "<end_of_turn>"
EOS = "<eos>"


def format_example(question: str, answer: str) -> str:
    return (
        f"{BOS}"
        f"{TURN_START}user\n{question}{TURN_END}\n"
        f"{TURN_START}model\n{answer}{TURN_END}"
        f"{EOS}"
    )


def load_pairs() -> list[dict]:
    if not IN_FILE.exists():
        print(f"[ERROR] Q&A pairs not found: {IN_FILE}")
        print("  Run step D (generate_qa_pipeline.py) first.")
        sys.exit(1)

    pairs = []
    seen_questions = set()
    skipped = 0
    duplicate_questions = 0

    with open(IN_FILE, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping malformed line {line_no}")
                skipped += 1
                continue

            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()
            category = (obj.get("category") or "general").strip() or "general"
            if not question or not answer or len(answer) < 30:
                skipped += 1
                continue
            if question in seen_questions:
                duplicate_questions += 1
                continue

            seen_questions.add(question)
            pairs.append({"question": question, "answer": answer, "category": category})

    print(
        f"  Loaded {len(pairs)} valid pairs "
        f"({skipped} skipped, {duplicate_questions} duplicate questions ignored)"
    )
    return pairs


def split(pairs: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
    random.seed(42)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(path: Path, pairs: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = {"text": format_example(pair["question"], pair["answer"])}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 60)
    print("  Gemma4GR Phase 2 - Prepare Q&A Dataset")
    print("=" * 60 + "\n")

    pairs = load_pairs()

    if len(pairs) < 50:
        print(f"[WARN] Only {len(pairs)} pairs. Recommend at least 200 for quality fine-tuning.")

    train_pairs, val_pairs = split(pairs, VAL_RATIO)
    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    write_jsonl(TRAIN_OUT, train_pairs)
    write_jsonl(VAL_OUT, val_pairs)

    cats: dict[str, int] = {}
    for pair in pairs:
        cats[pair["category"]] = cats.get(pair["category"], 0) + 1

    print("\n  Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda item: (-item[1], item[0])):
        bar = "#" * (count // 10)
        print(f"    {cat:<25} {count:4d}  {bar}")

    print("\n  Saved:")
    print(f"    {TRAIN_OUT}")
    print(f"    {VAL_OUT}")

    sample = train_pairs[0]
    text = format_example(sample["question"], sample["answer"])
    print("\n  Sample training record (first 200 chars):")
    print(f"  {text[:200]}...")

    print("\n  Dataset ready. Run step F to start training.\n")


if __name__ == "__main__":
    main()
