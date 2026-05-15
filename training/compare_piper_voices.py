"""
A/B Greek TTS: same texts through baseline Piper ONNX vs trained JOY ONNX.

Your ~300 human WAVs are **single-sentence prompts** from `data/voice_recording_manifest.csv`
(see `data/human_voice_dataset/metadata.csv`: id|text). They are not the Q&A JSONL.

This script synthesizes each chosen line twice (baseline voice, JOY voice) into WAVs for listening.

Sources (--source):
  qa            — first N rows of data/qa_pairs.jsonl (question + answer each row) [default]
  manifest      — first N texts from data/human_voice_dataset/metadata.csv (same sentences you recorded)

Env:
  PIPER_EXE              — piper.exe (default: GEMMA4KIDS_DIR/piper/piper.exe)
  PIPER_BASELINE_ONNX    — original Greek Piper voice (default: GEMMA4KIDS_DIR/voices/el_GR-rapunzelina-medium.onnx)
  PIPER_VOICE_ONNX       — trained JOY (default: output/piper_voice/el_GR-joy-medium.onnx)

Usage:
  python training/compare_piper_voices.py
  python training/compare_piper_voices.py --count 5 --source qa
  python training/compare_piper_voices.py --count 5 --source manifest --out-dir output/piper_ab_demo
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from console_encoding import ensure_utf8_console

ensure_utf8_console()

from dotenv import load_dotenv

from synthesize_qa_audio import _resolve_piper_exe, load_pairs

load_dotenv()

BASE = Path(__file__).parent.parent
METADATA_PATH = BASE / "data" / "human_voice_dataset" / "metadata.csv"
OUT_DEFAULT = BASE / "output" / "piper_voice_ab"


def _baseline_onnx() -> Path:
    raw = os.getenv("PIPER_BASELINE_ONNX", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (BASE / p).resolve()
    kid = os.getenv("GEMMA4KIDS_DIR", "").strip()
    if not kid:
        kid = str(BASE.parent / "Gemma4Kids")
    return Path(kid).expanduser() / "voices" / "el_GR-rapunzelina-medium.onnx"


def _joy_onnx() -> Path:
    raw = os.getenv("PIPER_VOICE_ONNX", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (BASE / p).resolve()
    return (BASE / "output" / "piper_voice" / "el_GR-joy-medium.onnx").resolve()


def _load_manifest_texts(n: int) -> list[tuple[str, str]]:
    """Return list of (label, text) for first n metadata lines."""
    if not METADATA_PATH.exists():
        print(f"[ERROR] {METADATA_PATH} not found.")
        sys.exit(1)
    rows: list[tuple[str, str]] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts[0].strip(), parts[1].strip()
            if not text:
                continue
            rows.append((utt_id, text))
            if len(rows) >= n:
                break
    return rows


def _synth(text: str, piper_exe: Path, model: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(piper_exe), "--model", str(model), "--output_file", str(out_wav)],
        input=text.encode("utf-8"),
        check=True,
        timeout=180,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs JOY Piper on the same Greek texts.")
    parser.add_argument("--count", type=int, default=5, help="Number of items (Q&A pairs or manifest lines)")
    parser.add_argument("--source", choices=("qa", "manifest"), default="qa")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DEFAULT.relative_to(BASE)),
        help="Project-relative or absolute output folder for WAVs",
    )
    args = parser.parse_args()

    piper_exe = _resolve_piper_exe()
    baseline = _baseline_onnx()
    joy = _joy_onnx()
    out_root = Path(args.out_dir).expanduser()
    if not out_root.is_absolute():
        out_root = (BASE / out_root).resolve()

    if not piper_exe.exists():
        print(f"[ERROR] Piper not found: {piper_exe}")
        sys.exit(1)
    if not baseline.exists():
        print(f"[ERROR] Baseline ONNX not found: {baseline}")
        print("  Set PIPER_BASELINE_ONNX or install Gemma4Kids Greek rapunzelina voice.")
        sys.exit(1)
    if not joy.exists():
        print(f"[ERROR] JOY ONNX not found: {joy}")
        sys.exit(1)

    jobs: list[tuple[str, str]] = []

    if args.source == "qa":
        pairs = load_pairs()
        n = max(1, min(args.count, len(pairs)))
        for i in range(n):
            p = pairs[i]
            jobs.append((f"pair{i:02d}_q", p["question"]))
            jobs.append((f"pair{i:02d}_a", p["answer"]))
    else:
        rows = _load_manifest_texts(max(1, args.count))
        if not rows:
            print("[ERROR] No manifest lines loaded.")
            sys.exit(1)
        for utt_id, text in rows:
            safe = utt_id.replace(" ", "_")
            jobs.append((safe, text))

    print(f"  Piper:    {piper_exe}")
    print(f"  Baseline: {baseline.name}")
    print(f"  JOY:      {joy.name}")
    print(f"  Out:      {out_root}")
    print(f"  Jobs:     {len(jobs)} clips x 2 voices\n")

    for stem, text in jobs:
        for label, model in (("baseline", baseline), ("joy", joy)):
            out = out_root / f"{stem}_{label}.wav"
            clip = text[:72] + ("..." if len(text) > 72 else "")
            print(f"  {out.relative_to(BASE)}  <=  ({label}) {clip}")
            _synth(text, piper_exe, model, out)

    print("\n  Done. Listen to matching stems: *_baseline.wav vs *_joy.wav")


if __name__ == "__main__":
    main()
