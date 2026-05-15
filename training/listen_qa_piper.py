"""
Play a few Greek Q&A lines through the trained Piper JOY voice (spot-check).

Reads data/qa_pairs.jsonl, runs piper.exe on each question and each answer,
plays WAVs in order using sounddevice.

Env (same as synthesize_qa_audio.py):
  PIPER_EXE, PIPER_VOICE_ONNX, GEMMA4KIDS_DIR

Usage:
  python training/listen_qa_piper.py
  python training/listen_qa_piper.py --count 3
  python training/listen_qa_piper.py --count 5 --save-dir output/piper_listen_demo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

from console_encoding import ensure_utf8_console

ensure_utf8_console()

from synthesize_qa_audio import _resolve_piper_exe, _resolve_voice_onnx, load_pairs

load_dotenv()

BASE = Path(__file__).parent.parent


def _piper_to_wav(text: str, piper_exe: Path, voice_onnx: Path, out_wav: Path) -> None:
    subprocess.run(
        [str(piper_exe), "--model", str(voice_onnx), "--output_file", str(out_wav)],
        input=text.encode("utf-8"),
        check=True,
        timeout=120,
    )


def _play_wav(path: Path) -> None:
    data, sr = sf.read(str(path), always_2d=False)
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1)
    sd.play(data, sr)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Listen to Piper TTS on first N Q&A pairs.")
    parser.add_argument("--count", type=int, default=3, help="Number of Q&A rows to play (default 3)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="If set, also copy generated WAVs here (project-relative or absolute)",
    )
    args = parser.parse_args()

    piper_exe = _resolve_piper_exe()
    voice_onnx = _resolve_voice_onnx()

    if not piper_exe.exists():
        print(f"[ERROR] Piper not found: {piper_exe}")
        print("  Run setup assets (menu 1) or set PIPER_EXE / GEMMA4KIDS_DIR in .env.")
        sys.exit(1)
    if not voice_onnx.exists():
        print(f"[ERROR] Voice ONNX not found: {voice_onnx}")
        print("  Train Piper (menu 9) or set PIPER_VOICE_ONNX in .env.")
        sys.exit(1)

    pairs = load_pairs()
    n = max(1, min(args.count, len(pairs)))
    save_root = Path(args.save_dir).resolve() if args.save_dir.strip() else None
    if save_root is not None and not save_root.is_absolute():
        save_root = (BASE / save_root).resolve()
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)

    print(f"  Piper: {piper_exe}")
    print(f"  Model: {voice_onnx}")
    print(f"  Playing first {n} pair(s) (question, then answer each).\n")

    for i in range(n):
        pair = pairs[i]
        q = pair["question"]
        a = pair["answer"]
        prefix = f"[{i + 1}/{n}]"

        for label, text in (("Q", q), ("A", a)):
            print(f"{prefix} {label}: {text[:120]}{'…' if len(text) > 120 else ''}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _piper_to_wav(text, piper_exe, voice_onnx, tmp_path)
                if save_root is not None:
                    ext = f"pair{i:03d}_{label.lower()}.wav"
                    dst = save_root / ext
                    dst.write_bytes(tmp_path.read_bytes())
                    try:
                        print(f"         saved {dst.relative_to(BASE)}")
                    except ValueError:
                        print(f"         saved {dst}")
                _play_wav(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            print()
    print("  Done.")


if __name__ == "__main__":
    main()
