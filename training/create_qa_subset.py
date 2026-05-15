"""
Create a deterministic subset of an existing Q&A JSONL corpus.

Defaults:
  Input:  data/qa_pairs.jsonl
  Output: data/qa_pairs_smoke_200.jsonl

Env vars:
  QA_DATASET_SOURCE   input JSONL relative to data/ or absolute path
  QA_SUBSET_OUTPUT    output JSONL relative to data/ or absolute path
  QA_SUBSET_LIMIT     number of pairs to keep
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from console_encoding import ensure_utf8_console

load_dotenv()
ensure_utf8_console()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"

_src_raw = os.getenv("QA_DATASET_SOURCE", "qa_pairs.jsonl").strip()
SOURCE = Path(_src_raw) if Path(_src_raw).is_absolute() else DATA_DIR / _src_raw
_out_raw = os.getenv("QA_SUBSET_OUTPUT", "qa_pairs_smoke_200.jsonl").strip()
OUTPUT = Path(_out_raw) if Path(_out_raw).is_absolute() else DATA_DIR / _out_raw
LIMIT = int(os.getenv("QA_SUBSET_LIMIT", "200"))
STRATEGY = os.getenv("QA_SUBSET_STRATEGY", "round_robin").strip().lower()


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with open(SOURCE, encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = (obj.get("question") or "").strip()
            answer = (obj.get("answer") or "").strip()
            category = (obj.get("category") or "general").strip() or "general"
            if not question or not answer:
                continue
            obj["category"] = category
            rows.append(obj)
    return rows


def pick_head(rows: list[dict]) -> list[dict]:
    return rows[:LIMIT]


def pick_round_robin(rows: list[dict]) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    category_order: list[str] = []

    for row in rows:
        category = row["category"]
        if category not in by_category:
            category_order.append(category)
        by_category[category].append(row)

    kept: list[dict] = []
    index_by_category = {category: 0 for category in category_order}

    while len(kept) < LIMIT:
        advanced = False
        for category in category_order:
            idx = index_by_category[category]
            bucket = by_category[category]
            if idx >= len(bucket):
                continue
            kept.append(bucket[idx])
            index_by_category[category] += 1
            advanced = True
            if len(kept) >= LIMIT:
                break
        if not advanced:
            break
    return kept


def print_breakdown(label: str, rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        cat = row["category"]
        counts[cat] = counts.get(cat, 0) + 1

    print(f"\n{label}:")
    for cat, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {cat:<22} {count:4d}")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"[ERROR] Source file not found: {SOURCE}")

    rows = load_rows()
    if STRATEGY == "head":
        subset = pick_head(rows)
    else:
        subset = pick_round_robin(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as dst:
        for obj in subset:
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(subset)} pairs -> {OUTPUT}")
    print(f"Strategy: {STRATEGY}")
    print_breakdown("Source category breakdown", rows)
    print_breakdown("Subset category breakdown", subset)


if __name__ == "__main__":
    main()
