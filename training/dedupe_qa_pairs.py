"""
Phase 2 - Cleanup Q&A pairs by deduplicating repeated questions.

Input:  data/qa_pairs.jsonl (or QA_DEDUPE_SOURCE)
Output: data/qa_pairs_deduped.jsonl (or QA_DEDUPE_OUTPUT)

The source file is preserved. For repeated questions, the script keeps one
record using a simple quality heuristic so downstream dataset prep can use a
clean corpus.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE = DATA_DIR / os.getenv("QA_DEDUPE_SOURCE", "qa_pairs.jsonl")
OUT_FILE = DATA_DIR / os.getenv("QA_DEDUPE_OUTPUT", "qa_pairs_deduped.jsonl")
MIN_ANSWER_LEN = int(os.getenv("QA_DEDUPE_MIN_ANSWER_LEN", "30"))


def answer_score(record: dict) -> tuple[int, int, int, int]:
    answer = (record.get("answer") or "").strip()
    category = (record.get("category") or "").strip()
    provider = (record.get("provider") or "").strip()
    model = (record.get("model") or "").strip()
    return (
        len(answer),
        1 if category else 0,
        1 if provider else 0,
        1 if model else 0,
    )


def load_records(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        print(f"[ERROR] Source file not found: {path}")
        sys.exit(1)

    records = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
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
            if not question or not answer or len(answer) < MIN_ANSWER_LEN:
                skipped += 1
                continue

            obj["question"] = question
            obj["answer"] = answer
            if "category" in obj and isinstance(obj["category"], str):
                obj["category"] = obj["category"].strip()
            records.append(obj)
    return records, skipped


def dedupe_records(records: list[dict]) -> tuple[list[dict], int]:
    best_by_question: dict[str, dict] = {}
    duplicate_rows = 0

    for record in records:
        question = record["question"]
        existing = best_by_question.get(question)
        if existing is None:
            best_by_question[question] = record
            continue

        duplicate_rows += 1
        if answer_score(record) > answer_score(existing):
            best_by_question[question] = record

    deduped = sorted(best_by_question.values(), key=lambda item: item["question"])
    return deduped, duplicate_rows


def write_records(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_top_duplicates(records: list[dict], top_n: int = 10) -> None:
    counts = Counter(record["question"] for record in records)
    repeated = [(question, count) for question, count in counts.most_common(top_n) if count > 1]
    if not repeated:
        return

    print("\n  Most repeated questions in source:")
    for question, count in repeated:
        print(f"    {count:2d}x {question}")


def main() -> None:
    print("=" * 60)
    print("  Gemma4GR Phase 2 - Dedupe Q&A Pairs")
    print(f"  Source: {IN_FILE}")
    print(f"  Output: {OUT_FILE}")
    print("=" * 60 + "\n")

    records, skipped = load_records(IN_FILE)
    total_rows = len(records)
    deduped, duplicate_rows = dedupe_records(records)

    write_records(OUT_FILE, deduped)

    print(f"  Valid source rows:   {total_rows}")
    print(f"  Unique questions:    {len(deduped)}")
    print(f"  Duplicate rows:      {duplicate_rows}")
    print(f"  Skipped bad rows:    {skipped}")
    print_top_duplicates(records)
    print("\n  Clean file written.")


if __name__ == "__main__":
    main()
