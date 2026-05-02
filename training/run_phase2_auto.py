"""
Phase 2 full-auto orchestrator.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from env_bootstrap import select_python_for_script

load_dotenv()

DATA_DIR = BASE / "data"

FULL_AUTO_MODE = os.getenv("PHASE2_AUTO_MODE", "full").strip().lower()
RUN_BASE_BENCH = os.getenv("PHASE2_AUTO_RUN_BASE_BENCH", "1").strip() == "1"
RUN_TRAIN = os.getenv("PHASE2_AUTO_RUN_TRAIN", "1").strip() == "1"
RUN_MERGE = os.getenv("PHASE2_AUTO_RUN_MERGE", "1").strip() == "1"
RUN_AFTER_BENCH = os.getenv("PHASE2_AUTO_RUN_AFTER_BENCH", "1").strip() == "1"
STOP_AFTER_DATA = os.getenv("PHASE2_AUTO_STOP_AFTER_DATA", "0").strip() == "1"

PREVIEW_ENABLED = os.getenv("PHASE2_AUTO_RUN_PREVIEW", "1").strip() == "1"
PREVIEW_REVIEW_REQUIRED = os.getenv("PHASE2_AUTO_REQUIRE_PREVIEW_REVIEW", "0").strip() == "1"
SMOKE_TEACHER_MODEL = os.getenv("PHASE2_AUTO_SMOKE_TEACHER_MODEL", "qwen3.5:9b").strip()

DEDUPED_SOURCE_NAME = os.getenv("QA_DEDUPE_OUTPUT", "qa_pairs_deduped.jsonl").strip()
DEDUPED_SOURCE_PATH = DATA_DIR / DEDUPED_SOURCE_NAME


def run_step(script: str, extra_env: dict[str, str] | None = None) -> None:
    script_path = BASE / script
    python_exe, reason = select_python_for_script(BASE, script_path)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"\n{'=' * 60}")
    print(f"[RUN] {script}")
    if reason:
        print(f"  Python: {reason}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [python_exe, str(script_path)],
        cwd=str(BASE),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def count_unique_questions(path: Path) -> int:
    if not path.exists():
        return 0

    questions = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = (row.get("question") or "").strip()
            if question:
                questions.add(question)
    return len(questions)


def configure_smoke_defaults() -> None:
    smoke_defaults = {
        "NUM_QA": "12",
        "QA_CHUNK_SIZE": "4",
        "QA_PREVIEW_COUNT": "4",
        "QA_BENCH_LIMIT": "4",
        "STT_BENCH_LIMIT": "2",
        "QA_TRAIN_MAX_SEQ_LEN": "1024",
        "QA_TRAIN_BATCH_SIZE": "1",
        "QA_TRAIN_GRAD_ACCUM": "1",
        "QA_TRAIN_EPOCHS": "1",
        "QA_TRAIN_MAX_STEPS": "1",
        "QA_TRAIN_SAVE_STEPS": "1",
        "QA_TRAIN_EVAL_STEPS": "1",
        "QA_TRAIN_LOGGING_STEPS": "1",
    }
    for key, value in smoke_defaults.items():
        os.environ.setdefault(key, value)
    if SMOKE_TEACHER_MODEL:
        os.environ["TEACHER_MODEL"] = SMOKE_TEACHER_MODEL


def maybe_run_preview() -> None:
    if not PREVIEW_ENABLED:
        return

    run_step(
        "training/generate_qa_pipeline.py",
        {
            "QA_MODE": "preview",
            "QA_OUTPUT_FILE": str(DATA_DIR / "qa_preview.jsonl"),
        },
    )

    if PREVIEW_REVIEW_REQUIRED:
        print("  [STOP] Preview review is required before bulk generation.")
        raise SystemExit(0)


def generate_until_target() -> None:
    target = int(os.environ.get("NUM_QA", "1500"))

    while True:
        current = count_unique_questions(DEDUPED_SOURCE_PATH)
        print(f"  Clean corpus progress: {current}/{target}")
        if current >= target:
            return

        run_step(
            "training/generate_qa_pipeline.py",
            {
                "QA_MODE": "full",
                "QA_OUTPUT_FILE": str(DEDUPED_SOURCE_PATH),
            },
        )

        new_total = count_unique_questions(DEDUPED_SOURCE_PATH)
        if new_total <= current:
            print("  [WARN] Generation did not increase the clean corpus.")
            print("  Stopping to avoid an endless loop. Check provider quality or question diversity.")
            return


def main() -> None:
    if FULL_AUTO_MODE == "smoke":
        configure_smoke_defaults()

    target = int(os.environ.get("NUM_QA", "1500"))

    print("=" * 60)
    print("  Gemma4GR Phase 2 - Full Auto")
    print(f"  Mode: {FULL_AUTO_MODE}")
    print(f"  Clean corpus target: {target}")
    print(f"  Clean corpus file:   {DEDUPED_SOURCE_PATH}")
    print("=" * 60)

    if FULL_AUTO_MODE == "smoke":
        print("  Smoke mode enabled: bounded generation, bounded benchmarks, one-step QA training.")
        if SMOKE_TEACHER_MODEL:
            print(f"  Smoke teacher override: {SMOKE_TEACHER_MODEL}")

    if RUN_BASE_BENCH:
        run_step("tests/run_before.py")

    maybe_run_preview()

    run_step("training/dedupe_qa_pairs.py")
    generate_until_target()
    run_step("training/dedupe_qa_pairs.py")
    run_step(
        "training/prepare_qa_dataset.py",
        {"QA_DATASET_SOURCE": DEDUPED_SOURCE_NAME},
    )
    run_step(
        "training/validate_samples.py",
        {"QA_DATASET_SOURCE": DEDUPED_SOURCE_NAME},
    )

    if STOP_AFTER_DATA:
        print("\n  Stopped after data pipeline by configuration.")
        return

    if RUN_TRAIN:
        run_step("training/train_qa_local.py")

    if RUN_MERGE:
        run_step("training/merge_adapters.py")

    if RUN_AFTER_BENCH:
        run_step("tests/run_after.py")

    print(f"\n{'=' * 60}")
    print("  Phase 2 full-auto run completed.")
    print(f"  Clean corpus: {DEDUPED_SOURCE_PATH}")
    print("  Train/val:    data/train_qa.jsonl , data/val_qa.jsonl")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
