"""
Step A — Run benchmarks on the FINE-TUNED model (after training).
Loads the trained E2B GGUF and runs the same benchmark suite.
"""
import os, sys, subprocess, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE       = Path(__file__).parent.parent
LLAMA_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))

# Fine-tuned GGUF
GGUF_E2B_FT = BASE / "output" / "e2b_greek_stt" / "gguf" / "gemma-4-E2B-it-Q4_K_M.gguf"
GGUF_E4B_FT = BASE / "output" / "e4b_greek_stt" / "gguf" / "gemma-4-E4B-it-Q4_K_M.gguf"

LLAMA_EXE = Path(os.getenv("GEMMA4KIDS_DIR",
    r"C:\Users\efso office\Desktop\Gemma4Kids")) / "release" / "win-unpacked" / "resources" / "llama-server.exe"

sys.path.insert(0, str(BASE / "tests"))
from greek_qa_suite import run_qa_benchmark
from stt_benchmark  import run_stt_benchmark


def find_llama_server() -> Path | None:
    if LLAMA_EXE.exists():
        return LLAMA_EXE
    return None


def pick_model() -> tuple[Path, str]:
    """Pick whichever fine-tuned model is available."""
    if GGUF_E4B_FT.exists():
        return GGUF_E4B_FT, "finetuned_e4b"
    if GGUF_E2B_FT.exists():
        return GGUF_E2B_FT, "finetuned"
    return None, None


def start_server(gguf: Path) -> subprocess.Popen | None:
    llama = find_llama_server()
    if not llama or not gguf:
        print("[WARN] llama-server or model not found — start server manually on port 8080.")
        return None

    print(f"  Starting llama-server with fine-tuned {gguf.name} ...")
    proc = subprocess.Popen(
        [str(llama), "--model", str(gguf), "--port", str(LLAMA_PORT),
         "--ctx-size", "4096", "--n-gpu-layers", "-1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        time.sleep(2)
        try:
            r = requests.get(f"http://localhost:{LLAMA_PORT}/health", timeout=2)
            if r.status_code == 200:
                print("  Server ready ✓")
                return proc
        except Exception:
            pass
    print("[WARN] Server did not respond — proceeding anyway.")
    return proc


def stop_server(proc):
    if proc:
        proc.terminate()


def main():
    print("=" * 55)
    print("  Gemma4GR — Benchmarks AFTER Training (Fine-tuned)")
    print("=" * 55 + "\n")

    model_path, model_label = pick_model()
    if model_path is None:
        print("[ERROR] No fine-tuned GGUF found.")
        print(f"  Expected: {GGUF_E2B_FT}")
        print("  Complete training (step 7 or 8) first.")
        sys.exit(1)

    print(f"  Using model: {model_path.name} (label: {model_label})")

    server = start_server(model_path)

    print("\n[1/2] Greek Q&A Benchmark ...")
    run_qa_benchmark(model_label)

    print("\n[2/2] STT Benchmark ...")
    run_stt_benchmark(model_label)

    stop_server(server)

    print("\n✓ Post-training benchmarks complete.")
    print("  Run option B in menu to compare before/after.")


if __name__ == "__main__":
    main()
