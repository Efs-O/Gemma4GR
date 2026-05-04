"""
Batch pitch-shift comparison using research-style presets (multiple semitones + librosa options).
Also tries pyrubberband once if rubberband-cli is available.

Output: data/raw_audio/pitch_shift_matrix/
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import soundfile as sf

BASE = Path(__file__).resolve().parent.parent


def _load_ps():
    spec = importlib.util.spec_from_file_location(
        "pitch_shift_wavs", BASE / "training" / "pitch_shift_wavs.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ps = _load_ps()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    inputs = [
        BASE / "data/raw_audio/pair_0000.wav",
        BASE / "data/raw_audio/pair_0004.wav",
    ]
    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        print("[ERROR] No pair_0000 / pair_0004 WAVs found.")
        sys.exit(1)

    out_root = BASE / "data/raw_audio/pitch_shift_matrix"
    out_root.mkdir(parents=True, exist_ok=True)

    # (label suffix, semitones, res_type, scale, n_fft, hop_length)
    librosa_presets: list[tuple[str, float, str, bool, int | None, int | None]] = [
        ("s2_soxr", 2.0, "soxr_hq", True, None, None),
        ("s3_soxr_sc", 3.0, "soxr_hq", True, None, None),
        ("s5_soxr_sc", 5.0, "soxr_hq", True, None, None),
        ("s3_kaiser_sc", 3.0, "kaiser_best", True, None, None),
        ("s5_kaiser_sc", 5.0, "kaiser_best", True, None, None),
        ("s5_s2048_h512", 5.0, "kaiser_best", True, 2048, 512),
    ]

    print(f"Output dir: {out_root}\n")

    for in_path in inputs:
        stem = in_path.stem
        y, sr = ps._load_mono_float32(in_path)
        for label, semi, rt, scale, nf, hop in librosa_presets:
            name = f"{stem}_{label}_p{semi:g}.wav"
            outp = out_root / name
            try:
                yo = ps.apply_pitch_shift(
                    y,
                    sr,
                    semi,
                    engine="librosa",
                    res_type=rt,
                    scale=scale,
                    n_fft=nf,
                    hop_length=hop,
                )
                sf.write(str(outp), yo, sr, subtype="PCM_16")
                print(f"  OK librosa  {name}")
            except Exception as e:
                print(f"  FAIL {name}: {e}")

        rb_out = out_root / f"{stem}_rb_s5_formant.wav"
        try:
            yo = ps.apply_pitch_shift(
                y, sr, 5.0, engine="rubberband", rubberband_voice=True
            )
            sf.write(str(rb_out), yo, sr, subtype="PCM_16")
            print(f"  OK rubberband {rb_out.name}")
        except Exception as e:
            print(f"  SKIP rubberband ({stem}): {e}")

    print("\nDone. Listen under data/raw_audio/pitch_shift_matrix/")


if __name__ == "__main__":
    main()
