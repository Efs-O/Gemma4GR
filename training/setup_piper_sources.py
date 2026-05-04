"""
Clone pinned Piper training dependencies into models/ (Docker build context + train mounts).

Repos stay gitignored; run once per machine before `docker build -f Dockerfile.piper`
and `training/train_piper.py`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

PHONEMIZE_URL = "https://github.com/rhasspy/piper-phonemize.git"
PIPER_URL = "https://github.com/rhasspy/piper.git"
PHONEMIZE_SHA = "ba3cc06c5248215928821f1393b2b854a936991a"
PIPER_SHA = "73c04d81d5590ecc46e522de3601ce7fb29fc2be"

DST_PHONEMIZE = MODELS / "piper-phonemize"
DST_PIPER = MODELS / "piper-src"


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def _head_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _sync_repo(dst: Path, url: str, rev: str) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not (dst / ".git").is_dir():
            raise SystemExit(f"[ERROR] {dst} exists but is not a git clone. Remove it and re-run.")
        if _head_sha(dst) == rev:
            print(f"  OK {dst.name} @ {rev[:8]}")
            return
        print(f"  Updating {dst.name} -> {rev[:8]}")
        _git("fetch", "origin", cwd=dst)
        _git("checkout", "-q", rev, cwd=dst)
        return

    print(f"  Cloning {url} -> {dst.name}")
    _git("clone", url, str(dst))
    _git("checkout", "-q", rev, cwd=dst)


def verify_layout() -> None:
    phon = DST_PHONEMIZE
    px = DST_PIPER / "src" / "python"
    for p, label in [(phon / "CMakeLists.txt", "piper-phonemize"), (px / "setup.py", "piper-src")]:
        if not p.exists():
            raise SystemExit(f"[ERROR] Missing {label} path: {p}")


def main() -> None:
    print("Gemma4GR — Piper source trees (pinned SHAs)")
    print(f"  Phonemize: {PHONEMIZE_SHA[:8]}…")
    print(f"  Piper:    {PIPER_SHA[:8]}…\n")
    if not shutil.which("git"):
        print("[ERROR] git not found in PATH.")
        sys.exit(1)
    try:
        _sync_repo(DST_PHONEMIZE, PHONEMIZE_URL, PHONEMIZE_SHA)
        _sync_repo(DST_PIPER, PIPER_URL, PIPER_SHA)
        verify_layout()
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] git command failed: {exc}")
        sys.exit(exc.returncode or 1)
    print("\n  Done. Build image: docker build -f Dockerfile.piper -t piper-training:local .")


if __name__ == "__main__":
    main()
