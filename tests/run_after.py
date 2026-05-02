"""
Step A: run post-training benchmarks on the fine-tuned model.

When GGUF export was skipped or failed softly, this script reports that state
from output/merge_summary.json and exits successfully instead of crashing the
smoke pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

BASE = Path(__file__).parent.parent
LLAMA_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_SERVER_EXE = os.getenv("LLAMA_SERVER_EXE", "")

MERGED_DIR = BASE / "output" / "merged_gguf"
MERGE_SUMMARY = BASE / "output" / "merge_summary.json"
GGUF_E2B_FT = BASE / "output" / "e2b_greek_stt" / "gguf" / "gemma-4-E2B-it-Q4_K_M.gguf"
GGUF_E4B_FT = BASE / "output" / "e4b_greek_stt" / "gguf" / "gemma-4-E4B-it-Q4_K_M.gguf"

LLAMA_EXE = Path(
    os.getenv("GEMMA4KIDS_DIR", r"C:\Users\efso office\Desktop\Gemma4Kids")
) / "release" / "win-unpacked" / "resources" / "llama-server.exe"

sys.path.insert(0, str(BASE / "tests"))
from greek_qa_suite import run_qa_benchmark
from stt_benchmark import run_stt_benchmark


def load_merge_summary() -> dict[str, object] | None:
    if not MERGE_SUMMARY.exists():
        return None

    try:
        return json.loads(MERGE_SUMMARY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def find_llama_server() -> Path | None:
    if LLAMA_SERVER_EXE:
        explicit = Path(LLAMA_SERVER_EXE)
        if explicit.exists():
            return explicit
    if LLAMA_EXE.exists():
        return LLAMA_EXE
    fallback = Path(r"C:\Program Files (x86)\Llamacpp\llama.cpp-b8929\llama-server.exe")
    if fallback.exists():
        return fallback
    return None


def pick_model() -> tuple[Path | None, str | None]:
    merged_q4 = next((f for f in MERGED_DIR.glob("*.gguf") if "q4" in f.name.lower()), None)
    if merged_q4:
        return merged_q4, "finetuned"
    if GGUF_E4B_FT.exists():
        return GGUF_E4B_FT, "finetuned_e4b"
    if GGUF_E2B_FT.exists():
        return GGUF_E2B_FT, "finetuned"
    return None, None


def start_server(gguf: Path) -> subprocess.Popen | None:
    llama = find_llama_server()
    if not llama or not gguf:
        print("[WARN] llama-server or model not found - start server manually on port 8080.")
        return None

    print(f"  Starting llama-server with fine-tuned {gguf.name} ...")
    proc = subprocess.Popen(
        [str(llama), "--model", str(gguf), "--port", str(LLAMA_PORT), "--ctx-size", "4096", "--n-gpu-layers", "-1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        time.sleep(2)
        try:
            response = requests.get(f"http://localhost:{LLAMA_PORT}/health", timeout=2)
            if response.status_code == 200:
                print("  Server ready")
                return proc
        except Exception:
            pass
    print("[WARN] Server did not respond - proceeding anyway.")
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc:
        proc.terminate()


def main() -> None:
    print("=" * 55)
    print("  Gemma4GR - Benchmarks After Training")
    print("=" * 55 + "\n")

    summary = load_merge_summary()
    if summary and summary.get("merge_completed") and not summary.get("gguf_exported"):
        print("  Merge completed, but no GGUF is available for llama.cpp benchmarking.")
        print(f"  Merged model artifacts: {summary.get('merged_model_dir')}")
        gguf_error = summary.get("gguf_error")
        if gguf_error:
            print(f"  GGUF export issue: {gguf_error}")
        print("  Skipping post-training benchmarks for this smoke run.")
        return

    model_path, model_label = pick_model()
    if model_path is None:
        print("[ERROR] No fine-tuned GGUF found.")
        print(f"  Expected merged GGUF under: {MERGED_DIR}")
        sys.exit(1)

    print(f"  Using model: {model_path.name} (label: {model_label})")

    server = start_server(model_path)
    try:
        print("\n[1/2] Greek Q&A benchmark ...")
        run_qa_benchmark(model_label)

        print("\n[2/2] STT benchmark ...")
        run_stt_benchmark(model_label)
    finally:
        stop_server(server)

    print("\nPost-training benchmarks complete.")
    print("  Run the before/after comparison next if needed.")


if __name__ == "__main__":
    main()
