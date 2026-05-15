"""Play every *.wav in a folder (sorted by name) — e.g. output from compare_piper_voices.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from console_encoding import ensure_utf8_console

ensure_utf8_console()

BASE = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Play WAV files in a folder in sorted order.")
    parser.add_argument(
        "--dir",
        type=str,
        default="output/piper_voice_ab",
        help="Folder with WAVs (project-relative or absolute)",
    )
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    if not folder.is_absolute():
        folder = (BASE / folder).resolve()
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"[ERROR] No WAV files in {folder}")
        sys.exit(1)
    print(f"  Playing {len(wavs)} file(s) from {folder}\n", flush=True)
    for path in wavs:
        print(f"  Now: {path.name}", flush=True)
        data, sr = sf.read(str(path), always_2d=False)
        if getattr(data, "ndim", 1) == 2:
            data = data.mean(axis=1)
        sd.play(data, sr)
        sd.wait()
    print("\n  Done.", flush=True)


if __name__ == "__main__":
    main()
