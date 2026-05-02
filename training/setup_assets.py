"""
Step 1 — Copy Piper runtime, Greek voice, and symlink GGUF models.
Run once before anything else.
"""
import os, shutil, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE        = Path(__file__).parent.parent
KIDS_DIR    = Path(os.getenv("GEMMA4KIDS_DIR", r"C:\Users\efso office\Desktop\Gemma4Kids"))
CACHE_DIR   = Path(os.getenv("GGUF_CACHE_DIR", r"N:\.cache\huggingface\hub"))

def copy_piper():
    src = KIDS_DIR / "piper"
    dst = BASE / "assets" / "piper"
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ["piper.exe", "onnxruntime.dll", "onnxruntime_providers_shared.dll"]:
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)
            print(f"  Copied: {fname}")
        else:
            print(f"  [WARN] Not found: {s}")

    espeak_src = src / "espeak-ng-data"
    espeak_dst = dst / "espeak-ng-data"
    if espeak_src.exists() and not espeak_dst.exists():
        shutil.copytree(espeak_src, espeak_dst)
        print("  Copied: espeak-ng-data/")
    elif espeak_dst.exists():
        print("  Skipped: espeak-ng-data/ (already present)")

def copy_voices():
    src = KIDS_DIR / "voices"
    dst = BASE / "assets" / "voices"
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ["el_GR-rapunzelina-medium.onnx", "el_GR-rapunzelina-medium.onnx.json"]:
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)
            print(f"  Copied: {fname}")
        else:
            print(f"  [WARN] Not found: {s}")

def find_gguf(model_slug: str) -> Path | None:
    """Find a Q4_K_M GGUF file for a given model slug in the HF cache."""
    pattern = f"models--unsloth--{model_slug}-GGUF"
    for candidate in CACHE_DIR.glob(f"{pattern}/snapshots/*/*.gguf"):
        if "Q4_K_M" in candidate.name or "q4_k_m" in candidate.name:
            return candidate
    # Fallback: any gguf
    for candidate in CACHE_DIR.glob(f"{pattern}/snapshots/*/*.gguf"):
        return candidate
    return None

def symlink_gguf():
    dst_dir = BASE / "assets" / "gguf"
    dst_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "gemma-4-E2B-it": "gemma-4-E2B-it-Q4_K_M.gguf",
        "gemma-4-E4B-it": "gemma-4-E4B-it-Q4_K_M.gguf",
    }

    for slug, link_name in models.items():
        src = find_gguf(slug)
        dst = dst_dir / link_name
        if dst.exists() or dst.is_symlink():
            print(f"  Skipped: {link_name} (already linked)")
            continue
        if src is None:
            print(f"  [WARN] GGUF not found for {slug} in {CACHE_DIR}")
            continue
        try:
            dst.symlink_to(src)
            print(f"  Linked: {link_name} → {src}")
        except (OSError, NotImplementedError):
            # Symlinks need Developer Mode or Admin on Windows — fall back to copy
            print(f"  [INFO] Symlink failed, copying instead (may take a while)...")
            shutil.copy2(src, dst)
            print(f"  Copied: {link_name}")

def check_env():
    if not (BASE / ".env").exists():
        print("[WARN] No .env file — copy .env.example → .env and set HF_TOKEN")
    else:
        from dotenv import dotenv_values
        cfg = dotenv_values(BASE / ".env")
        token = cfg.get("HF_TOKEN", "")
        if not token or token == "hf_your_token_here":
            print("[WARN] HF_TOKEN not set in .env")
        else:
            print(f"  HF_TOKEN: {token[:8]}...✓")

def main():
    print("=" * 55)
    print("  Gemma4GR — Asset Setup")
    print("=" * 55)

    print("\n[1/4] Checking .env ...")
    check_env()

    print("\n[2/4] Copying Piper runtime from Gemma4Kids ...")
    copy_piper()

    print("\n[3/4] Copying Greek voice models ...")
    copy_voices()

    print("\n[4/4] Linking GGUF models from cache ...")
    symlink_gguf()

    print("\n✓ Setup complete.")
    print(f"  Piper:  {BASE / 'assets/piper/piper.exe'}")
    print(f"  Voice:  {BASE / 'assets/voices/el_GR-rapunzelina-medium.onnx'}")
    print(f"  E2B:    {BASE / 'assets/gguf/gemma-4-E2B-it-Q4_K_M.gguf'}")
    print(f"  E4B:    {BASE / 'assets/gguf/gemma-4-E4B-it-Q4_K_M.gguf'}")

if __name__ == "__main__":
    main()
