"""
Step 2 — Download Moira GreekTTS-1.5 (Orpheus base + LoRA adapters) and SNAC codec.
Downloads ~6 GB total. Run once.
"""
import os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE        = Path(__file__).parent.parent
MOIRA_REPO  = os.getenv("MOIRA_REPO",    "moiralabs/GreekTTS-1.5")
ORPHEUS_BASE= os.getenv("ORPHEUS_BASE",  "unsloth/orpheus-3b-0.1-ft")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
MOIRA_DIR   = BASE / "models" / "moira"
ORPHEUS_DIR = BASE / "models" / "orpheus"

def check_token():
    if not HF_TOKEN or HF_TOKEN == "hf_your_token_here":
        print("[ERROR] HF_TOKEN not set. Edit .env and add your token.")
        sys.exit(1)
    print(f"  Token: {HF_TOKEN[:8]}...✓")

def download(repo_id: str, local_dir: Path, desc: str):
    local_dir.mkdir(parents=True, exist_ok=True)
    marker = local_dir / ".downloaded"
    if marker.exists():
        print(f"  Skipped: {desc} (already downloaded)")
        return

    print(f"  Downloading {desc} from {repo_id} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"],
    )
    marker.touch()
    print(f"  ✓ {desc} saved to {local_dir}")

def install_snac():
    print("  Checking snac package ...")
    try:
        import snac
        print("  snac already installed ✓")
    except ImportError:
        print("  Installing snac ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "snac", "-q"])
        print("  snac installed ✓")

def main():
    print("=" * 55)
    print("  Gemma4GR — Download Moira TTS Model")
    print("=" * 55)

    print("\n[1/4] Checking HF token ...")
    check_token()

    print("\n[2/4] Installing SNAC codec ...")
    install_snac()

    print("\n[3/4] Downloading Orpheus base model (~3 GB) ...")
    download(ORPHEUS_BASE, ORPHEUS_DIR, "Orpheus-3B base")

    print("\n[4/4] Downloading Moira GreekTTS-1.5 LoRA adapters ...")
    download(MOIRA_REPO, MOIRA_DIR, "Moira GreekTTS-1.5 adapters")

    print("\n✓ All models ready.")
    print(f"  Orpheus base: {ORPHEUS_DIR}")
    print(f"  Moira LoRA:   {MOIRA_DIR}")

if __name__ == "__main__":
    main()
