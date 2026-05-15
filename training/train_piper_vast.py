"""
Run Piper training directly inside a Vast.ai container.

This runner assumes the instance is already using the correct Piper image or an
equivalent provisioned environment. Unlike train_piper.py, it does not call
docker run. It executes Piper preprocess/train/export in the current container.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from console_encoding import ensure_utf8_console

ensure_utf8_console()
load_dotenv()

BASE = Path(__file__).parent.parent


def _piper_dataset_dir() -> Path:
    raw = os.getenv("PIPER_DATASET_DIR", "").strip()
    if not raw:
        return (BASE / "data" / "piper_dataset").resolve()
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (BASE / candidate).resolve()


DATASET_DIR = _piper_dataset_dir()
OUTPUT_DIR = BASE / "output" / "piper_voice"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
LIGHTNING_DIR = OUTPUT_DIR / "lightning_logs"
VALIDATION_STATUS = BASE / "logs" / "validation_status.json"

PIPER_BASE_CKPT = Path(os.getenv("PIPER_BASE_CKPT", "")).expanduser() if os.getenv("PIPER_BASE_CKPT") else None
PIPER_LANGUAGE = os.getenv("PIPER_LANGUAGE", "el")
PIPER_BATCH_SIZE = os.getenv("PIPER_BATCH_SIZE", "32")
PIPER_VALIDATION_SPLIT = os.getenv("PIPER_VALIDATION_SPLIT", "0.05")
PIPER_QUALITY = os.getenv("PIPER_QUALITY", "medium")


def _piper_max_epochs_str() -> str:
    return (os.getenv("PIPER_MAX_EPOCHS", "20") or "20").strip()


def warn_if_validation_stale(paths: list[Path], label: str) -> None:
    if not VALIDATION_STATUS.exists():
        print(f"  [WARN] No validation stamp found. Run step S (validate_samples.py) before {label}.")
        return

    try:
        status = json.loads(VALIDATION_STATUS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [WARN] Validation stamp is unreadable: {VALIDATION_STATUS}")
        return

    validated_files = status.get("files", {})
    stale = []
    for path in paths:
        if not path.exists():
            continue
        rel = str(path.relative_to(BASE))
        validated_mtime = validated_files.get(rel)
        if validated_mtime is None or path.stat().st_mtime > validated_mtime:
            stale.append(rel)

    if stale:
        print(f"  [WARN] Sample validation is stale for {label}. Re-run step S before a long training run.")
        for rel in stale:
            print(f"         Changed since validation: {rel}")


def check_remote_environment() -> None:
    print("  Python:", sys.version.split()[0])
    for exe in ("git",):
        if not shutil.which(exe):
            print(f"[ERROR] Required executable not found in PATH: {exe}")
            sys.exit(1)

    if shutil.which("docker"):
        print("  [WARN] docker is present, but this runner will not use nested docker.")

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(f"  Torch CUDA available: {cuda_ok}")
        if not cuda_ok:
            print("[ERROR] torch.cuda.is_available() is false.")
            sys.exit(1)
        print(f"  Torch version: {torch.__version__}")
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[ERROR] Failed to validate torch/CUDA environment: {exc}")
        sys.exit(1)

    try:
        import piper_train  # noqa: F401
        import piper_phonemize  # noqa: F401
    except Exception as exc:
        print(f"[ERROR] Piper packages are not importable in the current container: {exc}")
        sys.exit(1)


def check_dataset() -> int:
    meta = DATASET_DIR / "metadata.csv"
    wavs_dir = DATASET_DIR / "wavs"
    print(f"  Dataset directory: {DATASET_DIR}")
    if not meta.exists():
        print(f"[ERROR] {meta} not found.")
        sys.exit(1)
    if not wavs_dir.exists():
        print(f"[ERROR] {wavs_dir} not found.")
        sys.exit(1)

    with open(meta, encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]

    print(f"  Dataset entries: {len(rows)}")
    warn_if_validation_stale([meta], "Piper training")
    if len(rows) < 1000:
        print(f"  [WARN] Only {len(rows)} entries. 3000+ recommended for good quality.")

    bad_rows = []
    for row in rows[:20]:
        parts = row.split("|")
        if len(parts) != 2:
            bad_rows.append(row)
    if bad_rows:
        print("[ERROR] metadata.csv is not in single-speaker Piper format id|text")
        for row in bad_rows[:5]:
            print(f"  Bad row: {row}")
        sys.exit(1)

    return len(rows)


def check_checkpoint() -> None:
    if not PIPER_BASE_CKPT:
        print("[ERROR] PIPER_BASE_CKPT is not set in .env")
        sys.exit(1)
    if not PIPER_BASE_CKPT.exists():
        print(f"[ERROR] Piper checkpoint not found: {PIPER_BASE_CKPT}")
        sys.exit(1)
    if PIPER_BASE_CKPT.suffix != ".ckpt":
        print(f"[ERROR] Piper checkpoint must be a .ckpt file, got: {PIPER_BASE_CKPT.name}")
        sys.exit(1)
    print(f"  Base checkpoint: {PIPER_BASE_CKPT}")


def check_resume_vs_max_epochs() -> None:
    mx = int(_piper_max_epochs_str())
    path = PIPER_BASE_CKPT
    if path is None or not path.exists():
        return
    try:
        import torch
    except ImportError:
        print("  [WARN] torch missing - cannot verify PIPER_MAX_EPOCHS vs checkpoint epoch.")
        return
    try:
        ck = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        ep_raw = ck.get("epoch")
        if ep_raw is None:
            return
        cur = int(ep_raw) if not hasattr(ep_raw, "item") else int(ep_raw.item())
    except Exception as exc:
        print(f"  [WARN] Could not read checkpoint epoch: {exc}")
        return
    if mx <= cur:
        print("[ERROR] PIPER_MAX_EPOCHS must be greater than the checkpoint epoch when resuming.")
        print(f"       Checkpoint epoch: {cur}; PIPER_MAX_EPOCHS: {mx}")
        print(f"       Example: set PIPER_MAX_EPOCHS={cur + 5} or higher in .env")
        sys.exit(1)


def _run_and_stream(cmd: list[str], log_path: Path | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(BASE),
    )

    log_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, "w", encoding="utf-8")
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
    finally:
        if log_handle is not None:
            log_handle.close()

    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
        sys.exit(proc.returncode)


def preprocess_dataset() -> None:
    print("\n  Preprocessing Piper dataset ...")
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "piper_train.preprocess",
        "--language",
        PIPER_LANGUAGE,
        "--input-dir",
        str(DATASET_DIR),
        "--output-dir",
        str(PREPROCESSED_DIR),
        "--dataset-format",
        "ljspeech",
        "--single-speaker",
        "--sample-rate",
        "22050",
    ]
    _run_and_stream(cmd, BASE / "logs" / "piper_preprocess_vast.log")

    for needed in ("config.json", "dataset.jsonl"):
        if not (PREPROCESSED_DIR / needed).exists():
            print(f"[ERROR] Missing preprocess output: {PREPROCESSED_DIR / needed}")
            sys.exit(1)
    print("  Preprocess complete.")


def run_training() -> None:
    print("\n  Starting Piper training in the current container ...")
    print("  This runner is for Vast.ai / already-provisioned Linux containers.\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "piper_csv_launcher.py")),
        "--dataset-dir",
        str(PREPROCESSED_DIR),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--batch-size",
        PIPER_BATCH_SIZE,
        "--validation-split",
        PIPER_VALIDATION_SPLIT,
        "--num-test-examples",
        "0",
        "--max_epochs",
        _piper_max_epochs_str(),
        "--resume_from_checkpoint",
        str(PIPER_BASE_CKPT),
        "--checkpoint-epochs",
        "1",
        "--quality",
        PIPER_QUALITY,
        "--default_root_dir",
        str(OUTPUT_DIR),
    ]
    _run_and_stream(cmd, BASE / "logs" / "piper_training_vast.log")

    csv_dir = OUTPUT_DIR / "csv_metrics"
    if csv_dir.is_dir():
        print(f"\n  CSV metrics (loss curves): {csv_dir}")
        for m in sorted(csv_dir.rglob("metrics.csv")):
            print(f"    {m.relative_to(OUTPUT_DIR)}")


def export_onnx() -> None:
    print("\n  Exporting best checkpoint to ONNX ...")
    checkpoints = list(LIGHTNING_DIR.glob("version_*/checkpoints/*.ckpt"))
    if not checkpoints:
        print("[ERROR] No checkpoints found to export.")
        sys.exit(1)

    best = sorted(checkpoints)[-1]
    out_onnx = OUTPUT_DIR / "el_GR-joy-medium.onnx"
    out_json = OUTPUT_DIR / "el_GR-joy-medium.onnx.json"

    cmd = [
        sys.executable,
        "-m",
        "piper_train.export_onnx",
        str(best),
        str(out_onnx),
    ]
    _run_and_stream(cmd, BASE / "logs" / "piper_export_vast.log")

    shutil.copy2(PREPROCESSED_DIR / "config.json", out_json)
    print(f"  ONNX exported: {out_onnx}")
    print(f"  Config:        {out_json}")


def main() -> None:
    print("=" * 60)
    print("  Gemma4GR - Train Piper Greek Voice (Vast.ai)")
    print("=" * 60)
    print(f"  PIPER_DATASET_DIR -> {DATASET_DIR}\n")

    print("[1/4] Checking remote container environment ...")
    check_remote_environment()

    print("\n[2/4] Checking dataset + checkpoint ...")
    check_dataset()
    check_checkpoint()
    check_resume_vs_max_epochs()

    print("\n[3/4] Preprocessing dataset ...")
    preprocess_dataset()

    print("\n[4/4] Running training + export ...")
    run_training()
    export_onnx()


if __name__ == "__main__":
    main()
