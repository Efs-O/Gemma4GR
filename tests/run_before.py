"""
Step 6 — Run benchmarks on the BASE model (before fine-tuning).
Starts the llama.cpp server with E2B base model, runs all benchmarks,
then shuts down the server.
"""
import os, sys, subprocess, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE       = Path(__file__).parent.parent
LLAMA_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))

# Base model GGUF (before fine-tuning)
GGUF_E2B = BASE / "assets" / "gguf" / "gemma-4-E2B-it-Q4_K_M.gguf"

# Piper executable location
PIPER_EXE  = BASE / "assets" / "piper" / "piper.exe"
LLAMA_EXE  = Path(os.getenv("GEMMA4KIDS_DIR",
    r"C:\Users\efso office\Desktop\Gemma4Kids")) / "release" / "win-unpacked" / "resources" / "llama-server.exe"

sys.path.insert(0, str(BASE / "tests"))
from greek_qa_suite import run_qa_benchmark
from stt_benchmark  import run_stt_benchmark


def find_llama_server() -> Path | None:
    candidates = [
        LLAMA_EXE,
        Path(r"C:\Users\efso office\Desktop\Gemma4Kids\piper\llama-server.exe"),
        Path(r"C:\llama.cpp\llama-server.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def start_server(gguf: Path) -> subprocess.Popen | None:
    llama = find_llama_server()
    if not llama:
        print("[WARN] llama-server.exe not found — skipping server start.")
        print("  Start the server manually on port 8080 before running tests.")
        return None

    if not gguf.exists():
        print(f"[ERROR] Model not found: {gguf}")
        print("  Run step 1 (setup_assets.py) first.")
        return None

    print(f"  Starting llama-server with {gguf.name} ...")
    proc = subprocess.Popen(
        [str(llama), "--model", str(gguf), "--port", str(LLAMA_PORT),
         "--ctx-size", "4096", "--n-gpu-layers", "-1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    for _ in range(60):
        time.sleep(2)
        try:
            r = requests.get(f"http://localhost:{LLAMA_PORT}/health", timeout=2)
            if r.status_code == 200:
                print("  Server ready ✓")
                return proc
        except Exception:
            pass

    print("[WARN] Server did not respond in 120s — proceeding anyway.")
    return proc


def stop_server(proc: subprocess.Popen | None):
    if proc:
        proc.terminate()
        print("  Server stopped.")


def main():
    print("=" * 55)
    print("  Gemma4GR — Benchmarks BEFORE Training (Base Model)")
    print("=" * 55 + "\n")

    server = start_server(GGUF_E2B)

    print("\n[1/2] Greek Q&A Benchmark ...")
    run_qa_benchmark("base")

    print("\n[2/2] STT Benchmark ...")
    run_stt_benchmark("base")

    stop_server(server)

    print("\n✓ Pre-training benchmarks complete.")
    print("  Results saved to tests/benchmark_results/")
    print("  Run step A (run_after.py) after training to compare.")


if __name__ == "__main__":
    main()
