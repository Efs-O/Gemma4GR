"""
Build a fresh QA answers file from the latest fine-tuned E4B 4-bit GGUF.

Flow:
  1. Load the existing questions from data/qa_pairs.jsonl
  2. Start llama-server on the latest ft_e4b GGUF
  3. Ask the model each question with the same prompt/flags as compare_4bit_models_gui.py
  4. Save question + generated answer to a new isolated JSONL file

Output:
  data/e4b_self_refine/qa_pairs_e4b_generated.jsonl

After this file is created, you edit/fix the answers manually and then use
prepare_e4b_qa_finetune.py to turn it into train/val JSONL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
OUT_DIR = DATA_DIR / "e4b_self_refine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tests.compare_4bit_models_gui import (
    REQUEST_TIMEOUT,
    SHARED_CHAT_TEMPLATE,
    default_model_specs,
    start_server,
    stop_server,
    LLAMA_URL,
    SYSTEM_PROMPT,
    TEXT_PROMPT_INSTRUCTION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fresh E4B answers for existing QA questions.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Optional starting offset")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="",
        help="Optional source JSONL path. Default: data/qa_pairs.jsonl",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default="",
        help="Optional output JSONL path. Default: data/e4b_self_refine/qa_pairs_e4b_generated.jsonl",
    )
    return parser.parse_args()


def resolve_in_path(in_arg: str) -> Path:
    if in_arg.strip():
        path = Path(in_arg)
        return path if path.is_absolute() else BASE / path
    return DATA_DIR / "qa_pairs.jsonl"


def resolve_out_path(out_arg: str) -> Path:
    if out_arg.strip():
        path = Path(out_arg)
        return path if path.is_absolute() else BASE / path
    return OUT_DIR / "qa_pairs_e4b_generated.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
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


def request_text_answer_raw(question: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{TEXT_PROMPT_INSTRUCTION}\n\n{question}"},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 256,
    }
    response = requests.post(LLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if content:
        return content
    return (message.get("reasoning_content") or "").strip()


def load_questions(path: Path) -> list[dict]:
    rows = []
    for idx, row in enumerate(read_jsonl(path), start=1):
        question = (row.get("question") or "").strip()
        category = (row.get("category") or "general").strip() or "general"
        if not question:
            continue
        rows.append(
            {
                "id": f"qa_{idx:06d}",
                "category": category,
                "question": question,
            }
        )
    return rows


def latest_ft_e4b_spec():
    for spec in default_model_specs():
        if spec.key == "ft_e4b":
            return spec
    raise RuntimeError("ft_e4b model spec not found")


def start_ft_e4b_server(spec, log_dir: Path):
    return start_server(Path(spec.path), Path(spec.mmproj_path), str(SHARED_CHAT_TEMPLATE), log_dir)


def answer_with_recovery(question: str, spec, log_dir: Path, server):
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            answer = request_text_answer_raw(question)
            if answer:
                return answer, server
            raise RuntimeError("Empty answer from llama-server")
        except Exception as exc:
            last_error = exc
            stop_server(server)
            time.sleep(1.5)
            server = start_ft_e4b_server(spec, log_dir)
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    in_path = resolve_in_path(args.in_path)
    out_path = resolve_out_path(args.out_path)
    existing = {str(row.get("id")): row for row in read_jsonl(out_path) if row.get("id")}

    questions = load_questions(in_path)
    selected = questions[args.offset :]
    if args.limit > 0:
        selected = selected[: args.limit]

    spec = latest_ft_e4b_spec()
    log_dir = BASE / "output" / "e4b_self_refine" / "server_logs"
    server = None
    results: list[dict] = []

    print("=" * 72)
    print("Generate E4B QA answers")
    print(f"Source questions : {in_path}")
    print(f"Rows selected    : {len(selected)}")
    print(f"Output JSONL     : {out_path}")
    print(f"Model GGUF       : {spec.path}")
    print(f"mmproj           : {spec.mmproj_path}")
    print(f"chat template    : {SHARED_CHAT_TEMPLATE}")
    print("=" * 72)

    try:
        server = start_ft_e4b_server(spec, log_dir)

        for idx, row in enumerate(selected, start=1):
            row_id = row["id"]
            if row_id in existing:
                results.append(existing[row_id])
                continue

            print(f"[{idx}/{len(selected)}] {row_id} | {row['category']}")
            answer, server = answer_with_recovery(row["question"], spec, log_dir, server)
            results.append(
                {
                    "id": row_id,
                    "category": row["category"],
                    "question": row["question"],
                    "answer": answer,
                    "model_key": spec.key,
                    "model_label": spec.label,
                    "model_path": spec.path,
                }
            )

            if idx % 25 == 0:
                write_jsonl(out_path, results)
                print(f"  saved {idx} rows")

        write_jsonl(out_path, results)
        print(f"\nSaved: {out_path}")
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
