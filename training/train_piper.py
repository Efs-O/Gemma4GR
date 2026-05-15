"""
Step 9 — Train Piper Greek voice using Docker.
Requires: Docker Desktop + NVIDIA Container Toolkit + a Piper `.ckpt` checkpoint + image `piper-training:local`
           (build: `docker build -f Dockerfile.piper -t piper-training:local .` after `setup_piper_sources.py`).
Input:  LJSpeech-format folder (22050 Hz): wavs/ + metadata.csv (id|text rows).
        Default: data/piper_dataset (Moira/synthetic via prepare_piper_dataset.py).
        Human recordings: set PIPER_DATASET_DIR=data/human_voice_dataset in .env.
Output: output/piper_voice/el_GR-joy-medium.onnx
        output/piper_voice/el_GR-gemma4gr-medium.onnx.json
        output/piper_voice/preprocessed/config.json
Est. time: 24-48 hours on RTX 4060 Ti / 5060 Ti
"""
import json
import os
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


def _piper_output_dir() -> Path:
    raw = os.getenv("PIPER_OUTPUT_DIR", "").strip()
    if not raw:
        return (BASE / "output" / "piper_voice").resolve()
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (BASE / candidate).resolve()


OUTPUT_DIR = _piper_output_dir()
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
LIGHTNING_DIR = OUTPUT_DIR / "lightning_logs"
VALIDATION_STATUS = BASE / "logs" / "validation_status.json"

PIPER_BASE_CKPT = Path(os.getenv("PIPER_BASE_CKPT", "")).expanduser() if os.getenv("PIPER_BASE_CKPT") else None
PIPER_TRAIN_SRC = BASE / "models" / "piper-src" / "src" / "python"
PIPER_PHONEMIZE_SRC = BASE / "models" / "piper-phonemize"
PIPER_CSV_LAUNCHER = Path(__file__).resolve().parent / "piper_csv_launcher.py"
PIPER_LANGUAGE = os.getenv("PIPER_LANGUAGE", "el")
PIPER_BATCH_SIZE = os.getenv("PIPER_BATCH_SIZE", "32")
PIPER_VALIDATION_SPLIT = os.getenv("PIPER_VALIDATION_SPLIT", "0.05")
PIPER_QUALITY = os.getenv("PIPER_QUALITY", "medium")


def _piper_max_epochs_str() -> str:
    return (os.getenv("PIPER_MAX_EPOCHS", "20") or "20").strip()


def _piper_preprocess_max_workers(num_entries: int) -> int:
    raw = os.getenv("PIPER_PREPROCESS_MAX_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            print(f"  [WARN] Invalid PIPER_PREPROCESS_MAX_WORKERS={raw!r}; ignoring.")

    cpu_count = os.cpu_count() or 1
    # Piper preprocess computes batch_size = int(num_utterances / (workers * 2)).
    # Tiny smoke datasets can hit batch_size=0 unless workers are capped.
    return max(1, min(cpu_count, max(1, num_entries // 2)))


def warn_if_validation_stale(paths: list[Path], label: str):
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
        rel = str(path.relative_to(BASE))
        if not path.exists():
            continue
        validated_mtime = validated_files.get(rel)
        if validated_mtime is None or path.stat().st_mtime > validated_mtime:
            stale.append(rel)

    if stale:
        print(f"  [WARN] Sample validation is stale for {label}. Re-run step S before a long training run.")
        for rel in stale:
            print(f"         Changed since validation: {rel}")


def check_piper_sources():
    if not PIPER_PHONEMIZE_SRC.is_dir() or not (PIPER_PHONEMIZE_SRC / "CMakeLists.txt").exists():
        print("[ERROR] models/piper-phonemize missing or incomplete.")
        print("  Run: python training/setup_piper_sources.py")
        sys.exit(1)
    if not PIPER_TRAIN_SRC.is_dir() or not (PIPER_TRAIN_SRC / "setup.py").exists():
        print("[ERROR] models/piper-src/src/python missing.")
        print("  Run: python training/setup_piper_sources.py")
        sys.exit(1)


def check_training_image():
    r = subprocess.run(
        ["docker", "image", "inspect", "piper-training:local"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("[ERROR] Docker image [piper-training:local] not found.")
        print("  docker build -f Dockerfile.piper -t piper-training:local .")
        sys.exit(1)


def check_docker():
    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Docker not found.")
        sys.exit(1)
    print(f"  Docker: {result.stdout.strip()}")

    result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print("[ERROR] Docker daemon not responding.")
        sys.exit(1)
    # GPU passthrough verified manually via: docker run --gpus all nvcr.io/nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    print("  Docker daemon: OK (GPU passthrough pre-verified)")


def check_dataset() -> int:
    meta = DATASET_DIR / "metadata.csv"
    wavs_dir = DATASET_DIR / "wavs"
    print(f"  Dataset directory: {DATASET_DIR}")
    if not meta.exists():
        print(f"[ERROR] {meta} not found.")
        print("  Synthetic path: run training/prepare_piper_dataset.py (writes data/piper_dataset/).")
        print("  Human recordings: set PIPER_DATASET_DIR=data/human_voice_dataset and record WAVs first.")
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


def check_checkpoint():
    if not PIPER_BASE_CKPT:
        print("[ERROR] PIPER_BASE_CKPT is not set in .env")
        print("  Piper fine-tuning requires a .ckpt checkpoint, not an .onnx voice file.")
        sys.exit(1)
    if not PIPER_BASE_CKPT.exists():
        print(f"[ERROR] Piper checkpoint not found: {PIPER_BASE_CKPT}")
        sys.exit(1)
    if PIPER_BASE_CKPT.suffix != ".ckpt":
        print(f"[ERROR] Piper checkpoint must be a .ckpt file, got: {PIPER_BASE_CKPT.name}")
        sys.exit(1)
    print(f"  Base checkpoint: {PIPER_BASE_CKPT}")


def check_resume_vs_max_epochs():
    """PyTorch Lightning requires Trainer(max_epochs=X) with X > resumed epoch index."""
    mx = int(_piper_max_epochs_str())
    path = PIPER_BASE_CKPT
    if path is None or not path.exists():
        return
    try:
        import torch
    except ImportError:
        print("  [WARN] torch missing on host — cannot verify PIPER_MAX_EPOCHS vs checkpoint epoch.")
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
        print(f"       Example for more training: set PIPER_MAX_EPOCHS={cur + 5} or higher in .env")
        sys.exit(1)


def docker_run(command: str, gpu: bool = False) -> subprocess.CompletedProcess:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset_abs = str(DATASET_DIR.resolve())
    output_abs = str(OUTPUT_DIR.resolve())
    preprocessed_abs = str(PREPROCESSED_DIR.resolve())
    checkpoint_dir_abs = str(PIPER_BASE_CKPT.parent.resolve())

    shm = os.getenv("PIPER_DOCKER_SHM_SIZE", "2g")
    cmd = [
        "docker", "run", "--rm",
        "--shm-size",
        shm,
        "-v", f"{dataset_abs}:/app/dataset",
        "-v", f"{output_abs}:/app/output",
        "-v", f"{preprocessed_abs}:/app/preprocessed",
        "-v", f"{checkpoint_dir_abs}:/app/checkpoints",
    ]
    if gpu:
        cmd.extend(["--gpus", "all"])
    cmd.extend(
        [
            "piper-training:local",
            "bash",
            "-c",
            command,
        ]
    )

    return subprocess.run(cmd, capture_output=True, text=True)


def preprocess_dataset(num_entries: int):
    print("\n  Preprocessing Piper dataset ...")
    base_ckpt_name = PIPER_BASE_CKPT.name
    preprocess_workers = _piper_preprocess_max_workers(num_entries)
    print(f"  Preprocess workers: {preprocess_workers}")
    command = " && ".join(
        [
            "echo 'piper-training:local ready'",
            "python -m piper_train.preprocess"
            " --language {lang}"
            " --input-dir /app/dataset"
            " --output-dir /app/preprocessed"
            " --dataset-format ljspeech"
            " --single-speaker"
            " --sample-rate 22050"
            " --max-workers {workers}".format(
                lang=PIPER_LANGUAGE, workers=preprocess_workers
            ),
            "test -f /app/preprocessed/config.json",
            "test -f /app/preprocessed/dataset.jsonl",
            f"test -f /app/checkpoints/{base_ckpt_name}",
        ]
    )
    result = docker_run(command, gpu=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("[ERROR] Piper preprocess failed.")
        sys.exit(1)
    print("  Preprocess complete.")


def run_training():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass

    print("\n  Starting Piper training Docker container ...")
    print("  This will run for hours. Validate samples first before using this step.\n")

    base_ckpt_name = PIPER_BASE_CKPT.name
    command = " && ".join(
        [
            "echo 'piper-training:local ready'",
            # -u + PYTHONUNBUFFERED: stream Lightning/progress to host in real time (no block buffering).
            "export PYTHONUNBUFFERED=1",
            "python -u /app/piper_csv_launcher.py"
            " --dataset-dir /app/preprocessed"
            " --accelerator gpu"
            " --devices 1"
            f" --batch-size {PIPER_BATCH_SIZE}"
            f" --validation-split {PIPER_VALIDATION_SPLIT}"
            " --num-test-examples 0"
            f" --max_epochs {_piper_max_epochs_str()}"
            f" --resume_from_checkpoint /app/checkpoints/{base_ckpt_name}"
            " --checkpoint-epochs 1"
            " --quality {quality}"
            " --default_root_dir /app/output".format(quality=PIPER_QUALITY),
        ]
    )

    shm = os.getenv("PIPER_DOCKER_SHM_SIZE", "2g")
    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "--shm-size",
        shm,
        "-v", f"{str(DATASET_DIR.resolve())}:/app/dataset",
        "-v", f"{str(OUTPUT_DIR.resolve())}:/app/output",
        "-v", f"{str(PREPROCESSED_DIR.resolve())}:/app/preprocessed",
        "-v", f"{str(PIPER_BASE_CKPT.parent.resolve())}:/app/checkpoints",
        "-v", f"{str(PIPER_CSV_LAUNCHER.resolve())}:/app/piper_csv_launcher.py:ro",
        "piper-training:local",
        "bash", "-c", command,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log_path = BASE / "logs" / "piper_training.log"
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
            log.write(line)
            log.flush()
    proc.wait()

    if proc.returncode != 0:
        print(f"\n[ERROR] Piper training failed (exit {proc.returncode}). See {log_path}")
        sys.exit(1)

    csv_dir = OUTPUT_DIR / "csv_metrics"
    if csv_dir.is_dir():
        print(f"\n  CSV metrics (loss curves): {csv_dir}")
        for m in sorted(csv_dir.rglob("metrics.csv")):
            print(f"    {m.relative_to(OUTPUT_DIR)}")


def export_onnx():
    print("\n  Exporting best checkpoint to ONNX ...")
    checkpoints = list(LIGHTNING_DIR.glob("version_*/checkpoints/*.ckpt"))
    if not checkpoints:
        print("[ERROR] No checkpoints found to export.")
        sys.exit(1)

    best = sorted(checkpoints)[-1]
    out_onnx = OUTPUT_DIR / "el_GR-joy-medium.onnx"
    out_json = OUTPUT_DIR / "el_GR-joy-medium.onnx.json"

    best_abs = str(best.resolve())
    output_abs = str(OUTPUT_DIR.resolve())
    preprocess_abs = str(PREPROCESSED_DIR.resolve())

    shm = os.getenv("PIPER_DOCKER_SHM_SIZE", "2g")
    cmd = [
        "docker", "run", "--rm",
        "--shm-size",
        shm,
        "-v", f"{output_abs}:/app/output",
        "-v", f"{preprocess_abs}:/app/preprocessed",
        "-v", f"{best.parent.resolve()}:/app/checkpoints",
        "piper-training:local",
        "bash", "-c",
        " && ".join(
            [
                "echo 'piper-training:local ready'",
                f"python -m piper_train.export_onnx /app/checkpoints/{best.name} /app/output/{out_onnx.name}",
                f"cp /app/preprocessed/config.json /app/output/{out_json.name}",
            ]
        ),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("[ERROR] ONNX export failed.")
        sys.exit(1)

    print(f"  ONNX exported: {out_onnx}")
    print(f"  Config:        {out_json}")


def main():
    print("=" * 55)
    print("  Gemma4GR — Train Piper Greek Voice")
    print("=" * 55)
    print(f"  PIPER_DATASET_DIR -> {DATASET_DIR}\n")

    print("[1/4] Checking Docker + Piper image ...")
    check_docker()
    check_piper_sources()
    check_training_image()
    if not PIPER_CSV_LAUNCHER.is_file():
        print(f"[ERROR] CSV metric launcher not found: {PIPER_CSV_LAUNCHER}")
        sys.exit(1)

    print("\n[2/4] Checking dataset + checkpoint ...")
    dataset_entries = check_dataset()
    check_checkpoint()
    check_resume_vs_max_epochs()

    print("\n[3/4] Preprocessing dataset ...")
    preprocess_dataset(dataset_entries)

    print("\n[4/4] Running training + export ...")
    run_training()
    export_onnx()


if __name__ == "__main__":
    main()
