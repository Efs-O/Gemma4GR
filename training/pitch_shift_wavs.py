"""
Experimental post-process for Moira/Orpheus WAVs: pitch shift via librosa or pyrubberband.

librosa: phase-vocoder (--res-type, --scale, optional STFT size). Try smaller --semitones first
  (e.g. +2..+5); online guides often suggest ~+5 as a starting male-to-female nudge.

pyrubberband: needs rubberband-cli on PATH (Windows: install separately). Optional --formant
  / --pitch-hq via --rubberband-voice.

Examples:
  python training/pitch_shift_wavs.py --glob "data/raw_audio/pair_0000.wav" --semitones 3 --scale
  python training/pitch_shift_wavs.py --glob "*.wav" --semitones 5 --res-type kaiser_best --scale
  python training/pitch_shift_wavs.py --glob "x.wav" --engine rubberband --semitones 5 --rubberband-voice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _load_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, int(sr)


def apply_pitch_shift(
    y: np.ndarray,
    sr: int,
    semitones: float,
    *,
    engine: str = "librosa",
    res_type: str = "soxr_hq",
    scale: bool = False,
    n_fft: int | None = None,
    hop_length: int | None = None,
    rubberband_voice: bool = False,
) -> np.ndarray:
    if engine == "rubberband":
        import pyrubberband as pyrb

        rbargs: dict[str, str] | None = None
        if rubberband_voice:
            rbargs = {"--formant": "", "--pitch-hq": ""}
        return np.asarray(pyrb.pitch_shift(y, sr, semitones, rbargs=rbargs), dtype=np.float32)

    stft_kw: dict[str, int] = {}
    if n_fft is not None:
        stft_kw["n_fft"] = n_fft
    if hop_length is not None:
        stft_kw["hop_length"] = hop_length
    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=semitones,
        res_type=res_type,
        scale=scale,
        **stft_kw,
    )


def pitch_shift_file(
    in_path: Path,
    out_path: Path,
    semitones: float,
    *,
    engine: str = "librosa",
    res_type: str = "soxr_hq",
    scale: bool = False,
    n_fft: int | None = None,
    hop_length: int | None = None,
    rubberband_voice: bool = False,
) -> None:
    y, sr = _load_mono_float32(in_path)
    y_out = apply_pitch_shift(
        y,
        sr,
        semitones,
        engine=engine,
        res_type=res_type,
        scale=scale,
        n_fft=n_fft,
        hop_length=hop_length,
        rubberband_voice=rubberband_voice,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y_out, sr, subtype="PCM_16")


def main() -> None:
    p = argparse.ArgumentParser(description="Pitch-shift WAVs (experimental).")
    p.add_argument("--glob", required=True, help='e.g. data/raw_audio/pair_*.wav')
    p.add_argument("--semitones", type=float, default=3.0, help="Semitones (try +2..+5).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--suffix", type=str, default="_ps")
    p.add_argument("--engine", choices=("librosa", "rubberband"), default="librosa")
    p.add_argument(
        "--res-type",
        type=str,
        default="soxr_hq",
        help="librosa resampling after phase vocoder (e.g. soxr_hq, kaiser_best, kaiser_fast).",
    )
    p.add_argument(
        "--scale",
        action="store_true",
        help="librosa: match approximate energy of input.",
    )
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument(
        "--rubberband-voice",
        action="store_true",
        help="rubberband: pass --formant and --pitch-hq (voice-friendlier when CLI works).",
    )
    args = p.parse_args()

    base = Path(__file__).resolve().parent.parent
    glob_path = Path(args.glob)
    if not glob_path.is_absolute():
        glob_path = base / args.glob
    files = sorted(glob_path.parent.glob(glob_path.name))
    if not files:
        print(f"[ERROR] No files match: {glob_path}")
        sys.exit(1)

    outs: list[tuple[Path, Path]] = []
    for f in files:
        if f.suffix.lower() != ".wav":
            continue
        if args.out_dir is not None:
            out_p = Path(args.out_dir) / f"{f.stem}{args.suffix}.wav"
        else:
            out_p = f.parent / "pitch_shifted" / f"{f.stem}{args.suffix}.wav"
        outs.append((f, out_p))

    print(
        f"  Files: {len(outs)} | engine={args.engine} | semitones={args.semitones:+.2f} "
        f"| res_type={args.res_type} | scale={args.scale}"
    )
    for inp, outp in outs:
        print(f"  {inp.name} -> {outp}")
        try:
            pitch_shift_file(
                inp,
                outp,
                args.semitones,
                engine=args.engine,
                res_type=args.res_type,
                scale=args.scale,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                rubberband_voice=args.rubberband_voice,
            )
        except Exception as e:
            print(f"  [ERROR] {inp.name}: {e}")
            sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
