"""
Replace rows in the main E4B generated QA file with rows from a smoke file,
matched by `id`.

Default use:
  - source: data/e4b_self_refine/qa_pairs_e4b_generated_smoke10.jsonl
  - target: data/e4b_self_refine/qa_pairs_e4b_generated.jsonl

This is intended to repair the first 10 empty rows from the earlier failed run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair the main E4B generated QA file from a smoke file.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/e4b_self_refine/qa_pairs_e4b_generated_smoke10.jsonl",
        help="Replacement rows JSONL",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/e4b_self_refine/qa_pairs_e4b_generated.jsonl",
        help="Main generated JSONL to patch",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE / path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_path = resolve_path(args.source)
    target_path = resolve_path(args.target)

    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")
    if not target_path.exists():
        raise SystemExit(f"Target file not found: {target_path}")

    source_rows = read_jsonl(source_path)
    target_rows = read_jsonl(target_path)
    source_by_id = {str(row.get("id")): row for row in source_rows if row.get("id")}

    replaced = 0
    output_rows: list[dict] = []
    for row in target_rows:
        row_id = str(row.get("id") or "")
        if row_id and row_id in source_by_id:
            output_rows.append(source_by_id[row_id])
            replaced += 1
        else:
            output_rows.append(row)

    write_jsonl(target_path, output_rows)

    print("=" * 72)
    print("Repair complete")
    print(f"Source   : {source_path}")
    print(f"Target   : {target_path}")
    print(f"Replaced : {replaced}")
    print("=" * 72)


if __name__ == "__main__":
    main()
