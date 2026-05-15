"""
Prepare fine-tuning JSONL from a manually corrected E4B QA answers file.

Expected input:
  data/e4b_self_refine/qa_pairs_e4b_generated.jsonl

You manually fix the `answer` field in that file, then run this script.

Output:
  data/e4b_self_refine/qa_pairs_corrected_train.jsonl
  data/e4b_self_refine/qa_pairs_corrected_val.jsonl

Optional mode:
  --answers-only
  Use corrected answers as standalone sentence training items instead of QA pairs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
IN_DIR = DATA_DIR / "e4b_self_refine"
VAL_RATIO = 0.10

BOS = "<bos>"
TURN_START = "<start_of_turn>"
TURN_END = "<end_of_turn>"
EOS = "<eos>"
SYSTEM_PROMPT = (
    "Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. "
    "Μην χρησιμοποιείς άλλη γλώσσα εκτός αν ζητηθεί ρητά. "
    "Δώσε άμεση, ουσιαστική και πλήρη απάντηση."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare corrected E4B QA fine-tune dataset.")
    parser.add_argument("--answers-only", action="store_true", help="Train on corrected answers as sentences only")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="",
        help="Optional corrected JSONL path. Default: data/e4b_self_refine/qa_pairs_e4b_generated.jsonl",
    )
    return parser.parse_args()


def resolve_in_path(in_arg: str) -> Path:
    if in_arg.strip():
        path = Path(in_arg)
        return path if path.is_absolute() else BASE / path
    return IN_DIR / "qa_pairs_e4b_generated.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_qa(question: str, answer: str) -> str:
    return (
        f"{BOS}"
        f"{TURN_START}system\n{SYSTEM_PROMPT}{TURN_END}\n"
        f"{TURN_START}user\n{question}{TURN_END}\n"
        f"{TURN_START}model\n{answer}{TURN_END}"
        f"{EOS}"
    )


def format_answer_only(answer: str) -> str:
    return (
        f"{BOS}"
        f"{TURN_START}system\n{SYSTEM_PROMPT}{TURN_END}\n"
        f"{TURN_START}user\nΓράψε μία σωστή, φυσική ελληνική απάντηση.{TURN_END}\n"
        f"{TURN_START}model\n{answer}{TURN_END}"
        f"{EOS}"
    )


def build_records(rows: list[dict], answers_only: bool) -> list[dict]:
    records: list[dict] = []
    for row in rows:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not answer:
            continue
        if answers_only:
            text = format_answer_only(answer)
        else:
            if not question:
                continue
            text = format_qa(question, answer)
        records.append({"text": text})
    return records


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    random.seed(42)
    shuffled = records.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    args = parse_args()
    in_path = resolve_in_path(args.in_path)
    rows = read_jsonl(in_path)
    records = build_records(rows, args.answers_only)
    train_rows, val_rows = split_records(records)

    prefix = "qa_pairs_answers_only" if args.answers_only else "qa_pairs_corrected"
    train_path = IN_DIR / f"{prefix}_train.jsonl"
    val_path = IN_DIR / f"{prefix}_val.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print("=" * 72)
    print("Prepared E4B fine-tune dataset")
    print(f"Input      : {in_path}")
    print(f"Mode       : {'answers-only' if args.answers_only else 'qa-pairs'}")
    print(f"Examples   : {len(records)}")
    print(f"Train / Val: {len(train_rows)} / {len(val_rows)}")
    print(f"Saved      : {train_path}")
    print(f"Saved      : {val_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
