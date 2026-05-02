"""
Step 9 — Train Piper Greek voice using Docker.
Requires: Docker Desktop + NVIDIA Container Toolkit.
Input:  data/piper_dataset/ (LJSpeech format, 22050 Hz)
Output: output/piper_voice/el_GR-gemma4gr-medium.onnx
Est. time: 24-48 hours on RTX 4060 Ti / 5060 Ti
"""
import os, sys, subprocess, shutil, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE        = Path(__file__).parent.parent
DATASET_DIR = BASE / "data" / "piper_dataset"
OUTPUT_DIR  = BASE / "output" / "piper_voice"
VOICES_DIR  = BASE / "assets" / "voices"
BASE_VOICE  = "el_GR-rapunzelina-medium"


def check_docker():
    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] Docker not found.")
        print("  Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        sys.exit(1)
    print(f"  Docker: {result.stdout.strip()}")

    result = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8-base-ubuntu22.04", "nvidia-smi"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print("[ERROR] NVIDIA Container Toolkit not working.")
        print("  Guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
        sys.exit(1)
    print("  NVIDIA Container Toolkit: ✓")


def check_dataset():
    meta = DATASET_DIR / "metadata.csv"
    if not meta.exists():
        print(f"[ERROR] {meta} not found.")
        print("  Run step 5 (prepare_piper_dataset.py) first.")
        sys.exit(1)
    with open(meta, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  Dataset entries: {count}")
    if count < 1000:
        print(f"  [WARN] Only {count} entries. 3000+ recommended for good quality.")
    return count


def run_training():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve absolute paths for Docker volume mounts
    dataset_abs = str(DATASET_DIR.resolve())
    output_abs  = str(OUTPUT_DIR.resolve())
    voices_abs  = str(VOICES_DIR.resolve())

    # Base voice checkpoint path inside container
    base_ckpt = f"/pretrained/{BASE_VOICE}"

    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "-v", f"{dataset_abs}:/app/dataset",
        "-v", f"{output_abs}:/app/output",
        "-v", f"{voices_abs}:/pretrained",
        "nvcr.io/nvidia/pytorch:24.01-py3",
        "bash", "-c",
        " && ".join([
            "pip install piper-train -q",
            "python -m piper_train"
            f" --dataset-dir /app/dataset"
            " --accelerator gpu"
            " --devices 1"
            " --batch-size 32"
            " --validation-split 0.05"
            " --num-test-examples 0"
            " --max-epochs 20"
            f" --resume_from_checkpoint {base_ckpt}"
            " --checkpoint-epochs 1"
            " --lightning-dir /app/output/lightning_logs",
        ])
    ]

    print("\n  Starting Piper training Docker container ...")
    print("  This will run for 24-48 hours. You can safely close the menu")
    print("  and check logs/train_piper_*.log later.\n")

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
            print(line.rstrip())
            log.write(line)
    proc.wait()

    if proc.returncode != 0:
        print(f"\n[ERROR] Piper training failed (exit {proc.returncode}). See {log_path}")
        sys.exit(1)

    export_onnx()


def export_onnx():
    print("\n  Exporting best checkpoint to ONNX ...")
    lightning_dir = OUTPUT_DIR / "lightning_logs"
    checkpoints = list(lightning_dir.glob("version_*/checkpoints/*.ckpt"))

    if not checkpoints:
        print("[ERROR] No checkpoints found to export.")
        return

    best = sorted(checkpoints)[-1]
    out_onnx = OUTPUT_DIR / "el_GR-gemma4gr-medium.onnx"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{str(OUTPUT_DIR.resolve())}:/app/output",
        "nvcr.io/nvidia/pytorch:24.01-py3",
        "bash", "-c",
        f"pip install piper-train -q && "
        f"python -m piper_train.export_onnx "
        f"/app/output/lightning_logs/{best.relative_to(lightning_dir.parent)} "
        f"/app/output/el_GR-gemma4gr-medium.onnx"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        # Copy config from base voice
        src_json = VOICES_DIR / f"{BASE_VOICE}.onnx.json"
        dst_json = OUTPUT_DIR / "el_GR-gemma4gr-medium.onnx.json"
        if src_json.exists():
            config = json.loads(src_json.read_text(encoding="utf-8"))
            config["espeak"]["voice"] = "el"
            config["audio"]["sample_rate"] = 22050
            dst_json.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ONNX exported: {out_onnx}")
        print(f"  Config:        {dst_json}")
    else:
        print(f"  [WARN] ONNX export failed: {result.stderr}")


def main():
    print("=" * 55)
    print("  Gemma4GR — Train Piper Greek Voice")
    print("=" * 55 + "\n")

    print("[1/3] Checking Docker + NVIDIA ...")
    check_docker()

    print("\n[2/3] Checking dataset ...")
    check_dataset()

    print("\n[3/3] Running training ...")
    run_training()


if __name__ == "__main__":
    main()
