"""
Final local training orchestrator.

Runs the selected local jobs sequentially with a shared output root, so the
last training pass is reproducible and saved off the system drive.

Default jobs (E4B only):
  - rebuild_stt_final_data -- rebuild the strict STT-only final dataset
  - e4b_qa                 -- text Q&A LoRA on E4B (train_qa_combined.jsonl)
  - e4b_stt_final          -- strict STT-only LoRA on E4B
  - e4b_stt_final_merge    -- merge the QA adapter with the final STT adapter

Optional rebuild jobs:
  - rebuild_stt_qa_data    -- synthesize QA question WAVs and rebuild
                              train_stt_qa.jsonl / val_stt_qa.jsonl
  - rebuild_stt_final_data -- rebuild train_stt_final.jsonl / val_stt_final.jsonl

E2B jobs are still available via FINAL_RUN_JOBS env var:
  FINAL_RUN_JOBS=e2b_qa,e2b_stt_qa,e2b_merge

To run the older mixed STT-QA path, use:
  FINAL_RUN_JOBS=rebuild_stt_qa_data,e4b_qa,e4b_stt_qa,e4b_merge

Env overrides:
  FINAL_RUN_JOBS=e2b_qa,e2b_stt_qa,...   run only selected jobs
  STT_QA_SKIP_QA_AUDIO=1                 omit JOY QA audio from STT dataset
  FINAL_RUN_REPLACE_EXISTING=0           keep existing adapter checkpoints
  FINAL_RUN_DRY_RUN=1                    print commands without running
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from env_bootstrap import normalize_hf_model_path, select_python_for_script

load_dotenv()

DEFAULT_OUTPUT_BASE = Path(r"N:\Gemma4GR\training_output")
E4B_MODEL_CACHE = Path(r"N:\.cache\huggingface\hub\models--unsloth--gemma-4-E4B-it")
OUTPUT_ROOT_RAW = os.getenv("GEMMA4GR_OUTPUT_ROOT", "").strip()
if OUTPUT_ROOT_RAW:
    OUTPUT_ROOT = Path(OUTPUT_ROOT_RAW)
    if not OUTPUT_ROOT.is_absolute():
        OUTPUT_ROOT = BASE / OUTPUT_ROOT
else:
    SESSION = os.getenv("TRAINING_SESSION", "").strip() or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    OUTPUT_ROOT = DEFAULT_OUTPUT_BASE / SESSION

FINAL_RUN_JOBS = [
    job.strip().lower()
    for job in os.getenv(
        "FINAL_RUN_JOBS",
        "rebuild_stt_final_data,e4b_qa,e4b_stt_final,e4b_stt_final_merge",
    ).split(",")
    if job.strip()
]
REPLACE_EXISTING = os.getenv("FINAL_RUN_REPLACE_EXISTING", "1").strip().lower() in ("1", "true", "yes")
DRY_RUN = os.getenv("FINAL_RUN_DRY_RUN", "0").strip().lower() in ("1", "true", "yes")


def e4b_model_path() -> str:
    return normalize_hf_model_path(str(E4B_MODEL_CACHE)) or str(E4B_MODEL_CACHE)


def run_python(script: str, extra_env: dict[str, str] | None = None) -> None:
    script_path = BASE / script
    python_exe, _ = select_python_for_script(BASE, script_path)
    env = os.environ.copy()
    env["GEMMA4GR_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    if extra_env:
        env.update(extra_env)

    print(f"\n{'=' * 72}")
    print(f"[RUN] {script}")
    print(f"  output root: {OUTPUT_ROOT}")
    print(f"{'=' * 72}\n")

    if DRY_RUN:
        if extra_env:
            for key, value in sorted(extra_env.items()):
                print(f"  {key}={value}")
        return

    result = subprocess.run(
        [python_exe, str(script_path)],
        cwd=str(BASE),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def clear_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


_GGUF_CACHE = Path(os.environ.get("GGUF_CACHE_DIR", r"N:\.cache\huggingface\hub"))


def prepare_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Each run uses its own session folder; nothing to clear by default.
    if not REPLACE_EXISTING or not OUTPUT_ROOT_RAW:
        return

    targets = [
        OUTPUT_ROOT / "e2b_greek_qa",
        OUTPUT_ROOT / "e2b_stt_qa",
        OUTPUT_ROOT / "e4b_greek_qa",
        OUTPUT_ROOT / "e4b_stt_qa",
        OUTPUT_ROOT / "e4b_stt_final",
        OUTPUT_ROOT / "merged_model",
        OUTPUT_ROOT / "merge_summary_e2b.json",
        OUTPUT_ROOT / "merge_summary_e4b.json",
        _GGUF_CACHE / "gemma-4-E2B-it-GR",
        _GGUF_CACHE / "gemma-4-E4B-it-GR",
        _GGUF_CACHE / "gemma4gr-e2b",
        _GGUF_CACHE / "gemma4gr-e2b_gguf",
        _GGUF_CACHE / "gemma4gr-e4b",
        _GGUF_CACHE / "gemma4gr-e4b_gguf",
    ]
    for target in targets:
        if clear_path(target):
            print(f"[CLEARED] {target}")


def run_job(job: str) -> None:
    if job == "rebuild_stt_final_data":
        run_python("training/prepare_stt_final_dataset.py")
        return

    if job == "rebuild_stt_qa_data":
        run_python("training/synthesize_qa_audio.py")
        run_python("training/prepare_stt_qa_dataset.py")
        return

    if job == "e2b_qa":
        run_python(
            "training/train_qa_local.py",
            {
                "E2B_MODEL": "unsloth/gemma-4-E2B-it",
                "QA_OUTPUT_DIR": str(OUTPUT_ROOT / "e2b_greek_qa"),
            },
        )
        return

    if job == "e2b_stt_qa":
        run_python(
            "training/train_stt_qa_local.py",
            {
                "STT_QA_MODEL": "unsloth/gemma-4-E2B-it",
                "STT_QA_OUTPUT_DIR": str(OUTPUT_ROOT / "e2b_stt_qa"),
            },
        )
        return

    if job == "e2b_merge":
        run_python(
            "training/merge_adapters.py",
            {
                "MERGE_MODEL": "e2b",
            },
        )
        return

    if job == "e4b_qa":
        run_python(
            "training/train_qa_local.py",
            {
                "E2B_MODEL": "unsloth/gemma-4-E4B-it",
                "E2B_MODEL_PATH": e4b_model_path(),
                "QA_OUTPUT_DIR": str(OUTPUT_ROOT / "e4b_greek_qa"),
                "QA_TRAIN_DATA": str(BASE / "data" / "train_qa_combined.jsonl"),
            },
        )
        return

    if job == "e4b_stt_qa":
        run_python(
            "training/train_stt_qa_local.py",
            {
                "STT_QA_MODEL": "unsloth/gemma-4-E4B-it",
                "STT_QA_MODEL_PATH": e4b_model_path(),
                "STT_QA_OUTPUT_DIR": str(OUTPUT_ROOT / "e4b_stt_qa"),
            },
        )
        return

    if job == "e4b_stt_final":
        run_python(
            "training/train_stt_final_local.py",
            {
                "STT_FINAL_MODEL": "unsloth/gemma-4-E4B-it",
                "STT_FINAL_MODEL_PATH": e4b_model_path(),
                "STT_FINAL_OUTPUT_DIR": str(OUTPUT_ROOT / "e4b_stt_final"),
            },
        )
        return

    if job == "e4b_merge":
        run_python(
            "training/merge_adapters.py",
            {
                "MERGE_MODEL": "e4b",
            },
        )
        return

    if job == "e4b_stt_final_merge":
        run_python(
            "training/merge_adapters.py",
            {
                "MERGE_MODEL": "e4b",
                "MERGE_STT_ADAPTER_DIR": str(OUTPUT_ROOT / "e4b_stt_final" / "lora_adapter"),
            },
        )
        return

    raise SystemExit(f"Unknown FINAL_RUN_JOBS entry: {job}")


def main() -> None:
    print("=" * 72)
    print("Gemma4GR final local training sequence")
    print(f"Session     : {OUTPUT_ROOT.name}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Jobs        : {', '.join(FINAL_RUN_JOBS)}")
    print(f"Dry run     : {DRY_RUN}")
    print("=" * 72)

    prepare_output_root()

    for job in FINAL_RUN_JOBS:
        run_job(job)

    print(f"\n{'=' * 72}")
    print("Final local training sequence completed.")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
