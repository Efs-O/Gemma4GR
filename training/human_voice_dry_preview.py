"""
Pick a handful of human voice WAVs, score likely room/tail / dull-tail issues,
copy originals to a preview folder, and write aggressively-labeled "cleaned"
copies using only numpy/scipy/librosa/soundfile (no extra pip deps).

Detection is heuristic (tail energy ratio, reflection-ish autocorr in tail,
spectral centroid vs. peak level as a weak "hollow/distant" proxy). It does
NOT classify latency, headphone bleed, or specific plugin chains.

Processing reduces diffuse energy (spectral subtraction toward a quiet-frame
noise profile, high-pass, optional gentle re-tilt when centroid is very low).
It may thin consonants if pushed; defaults stay conservative.

Usage:
  python training/human_voice_dry_preview.py
  python training/human_voice_dry_preview.py --source data/human_voice_dataset/wavs
  python training/human_voice_dry_preview.py --count 5 --highpass-hz 100
  python training/human_voice_dry_preview.py --all --batch-out data/human_voice_dataset/wavs_dry
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

BASE = Path(__file__).resolve().parent.parent

FRAME = 1024
HOP_F = 256
THR_MULT = 4.0
RMS_WIN = 1024
RMS_HOP = 256


def _framed_rms(y: np.ndarray) -> np.ndarray:
    if len(y) < FRAME:
        return np.array([float(np.sqrt(np.mean(y**2)))], dtype=np.float64)
    n = 1 + (len(y) - FRAME) // HOP_F
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sl = i * HOP_F
        out[i] = np.sqrt(np.mean(y[sl : sl + FRAME] ** 2))
    return out


def _autocorr_reflection_peak(x: np.ndarray, sr: int, min_ms: float = 12, max_ms: float = 250) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    nrm = float(np.linalg.norm(x))
    if nrm < 1e-9 or len(x) < int(0.05 * sr):
        return 0.0
    x = x / nrm
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2 :]
    if ac[0] <= 1e-12:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr * min_ms / 1000.0)
    hi = min(int(sr * max_ms / 1000.0), len(ac) - 1)
    if lo >= hi:
        return 0.0
    return float(np.max(ac[lo : hi + 1]))


def _score_clip(path: Path) -> dict[str, Any]:
    y, sr = sf.read(path, always_2d=False)
    if getattr(y, "ndim", 1) == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    peak = float(np.max(np.abs(y))) + 1e-12
    peak_dbfs = 20.0 * math.log10(peak + 1e-12)
    rms = _framed_rms(y)
    q15 = float(np.percentile(rms, 15))
    mx = float(np.max(rms))
    thr = max(q15 * THR_MULT, mx * 0.05)
    active = rms > thr
    if not np.any(active):
        return {
            "path": str(path.relative_to(BASE)),
            "tail_ratio": 0.0,
            "refl_peak": 0.0,
            "spectral_proxy": 0.0,
            "composite": 0.0,
            "note": "no_active_frames",
        }
    last_a = int(np.where(active)[0][-1])
    speech_e = float(np.sum(rms[active] ** 2))
    tail_rms = rms[last_a + 1 :]
    tail_e = float(np.sum(tail_rms**2)) if len(tail_rms) else 0.0
    tail_ratio = tail_e / (speech_e + 1e-12)
    start_samp = min((last_a + 1) * HOP_F, max(len(y) - 1, 0))
    tail_audio = y[start_samp:]
    refl = _autocorr_reflection_peak(tail_audio, sr)

    cent = float(np.median(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=RMS_WIN, hop_length=RMS_HOP)))
    dull = max(0.0, (4200.0 - cent) / 4200.0)
    quiet_proxy = max(0.0, min(1.0, (-24.0 - peak_dbfs) / 18.0))
    spec_proxy = round(min(1.0, 0.65 * dull + 0.35 * quiet_proxy), 4)

    comp = (
        120.0 * min(tail_ratio, 0.25)
        + 0.9 * refl
        + 0.35 * spec_proxy
    )
    return {
        "path": str(path.relative_to(BASE)),
        "tail_ratio": round(tail_ratio, 6),
        "refl_peak": round(refl, 4),
        "spectral_proxy": round(spec_proxy, 4),
        "composite": round(comp, 4),
        "sr": sr,
        "duration_s": round(len(y) / float(sr), 3),
    }


def _highpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if hz <= 0:
        return y
    ny = sr * 0.5
    w = hz / ny
    w = min(w, 0.99)
    b, a = signal.butter(2, w, btype="high")
    return signal.filtfilt(b, a, y)


def _spectral_subtract(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    hop: int = 256,
    alpha: float = 1.8,
    beta_floor: float = 0.12,
    tail_extra: float = 0.85,
) -> np.ndarray:
    y = y.astype(np.float64, copy=False)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)
    frame_e = np.mean(mag**2, axis=0)
    q10 = np.percentile(frame_e, 10)
    noise_mask = frame_e <= q10 * 1.6
    if not np.any(noise_mask):
        noise_mask = frame_e <= np.percentile(frame_e, 5)
    noise_prof = np.median(mag[:, noise_mask], axis=1, keepdims=True)
    cleaned = mag - alpha * noise_prof
    cleaned = np.maximum(cleaned, beta_floor * mag)
    mx = float(np.max(frame_e)) + 1e-12
    thr = max(float(np.percentile(frame_e, 15)) * 4.0, mx * 0.05)
    active = frame_e > thr
    if np.any(active):
        last_a = int(np.where(active)[0][-1])
        tail_cols = np.arange(mag.shape[1]) > last_a
        cleaned[:, tail_cols] *= tail_extra
    S_out = cleaned * np.exp(1j * phase)
    y_out = librosa.istft(S_out, hop_length=hop, length=len(y))
    return y_out.astype(np.float64)


def _to_int16_write(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    m = float(np.max(np.abs(y))) + 1e-12
    if m <= 1.5:
        return np.clip(np.rint(y * 32767.0), -32768, 32767).astype(np.int16)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def _norm_peak_match(y_in: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    p_ref = float(np.max(np.abs(y_ref))) + 1e-12
    p_out = float(np.max(np.abs(y_in))) + 1e-12
    return y_in * (p_ref / p_out)


def _process_one(
    y: np.ndarray,
    sr: int,
    *,
    highpass_hz: float,
    sub_alpha: float,
    tail_atten: float,
) -> np.ndarray:
    y = y.astype(np.float64, copy=False)
    y = _highpass(y, sr, highpass_hz)
    y = _spectral_subtract(y, sr, alpha=sub_alpha, tail_extra=tail_atten)
    cent_after = float(np.median(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=RMS_WIN, hop_length=RMS_HOP)))
    if cent_after < 2800:
        y = librosa.effects.preemphasis(y, coef=0.97)
    return y


def _pick_examples(scores: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    by_tail = sorted(scores, key=lambda d: d.get("tail_ratio", 0.0), reverse=True)
    by_refl = sorted(scores, key=lambda d: d.get("refl_peak", 0.0), reverse=True)
    by_comp = sorted(scores, key=lambda d: d.get("composite", 0.0), reverse=True)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(row: dict[str, Any]) -> None:
        p = row["path"]
        if p not in seen and len(chosen) < k:
            seen.add(p)
            chosen.append(row)

    take(by_comp[0])
    take(by_tail[0])
    take(by_refl[0])
    i = 1
    while len(chosen) < k and i < len(by_tail):
        take(by_tail[i])
        i += 1
    i = 1
    while len(chosen) < k and i < len(by_refl):
        take(by_refl[i])
        i += 1
    mid = sorted(scores, key=lambda d: d.get("composite", 0.0))[len(scores) // 2]
    take(mid)
    j = 0
    while len(chosen) < k and j < len(by_comp):
        take(by_comp[j])
        j += 1
    return chosen[:k]


def main() -> None:
    p = argparse.ArgumentParser(description="Dry-preview: score, copy, lightly process 5 WAVs.")
    p.add_argument("--source", type=str, default="data/human_voice_dataset/wavs")
    p.add_argument("--out", type=str, default="data/human_voice_dataset/dry_preview")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--all", action="store_true", help="Process every WAV in source into --batch-out")
    p.add_argument(
        "--batch-out",
        type=str,
        default="data/human_voice_dataset/wavs_dry",
        help="Output directory when using --all",
    )
    p.add_argument("--highpass-hz", type=float, default=90.0)
    p.add_argument("--sub-alpha", type=float, default=1.8, help="spectral subtraction strength")
    p.add_argument("--tail-atten", type=float, default=0.88, help="extra mag scale on STFT tail cols")
    args = p.parse_args()

    src_dir = (BASE / args.source).resolve()
    wavs = sorted(src_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAV files under {src_dir}")

    if args.all:
        batch_dir = (BASE / args.batch_out).resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": "all",
            "source": str(src_dir.relative_to(BASE)),
            "params": {
                "highpass_hz": args.highpass_hz,
                "sub_alpha": args.sub_alpha,
                "tail_atten": args.tail_atten,
            },
            "files": [],
        }
        for i, wav_path in enumerate(wavs):
            y, sr = sf.read(wav_path, always_2d=False)
            if getattr(y, "ndim", 1) == 2:
                y = y.mean(axis=1)
            y = np.asarray(y, dtype=np.float64)
            y2 = _process_one(
                y,
                sr,
                highpass_hz=args.highpass_hz,
                sub_alpha=args.sub_alpha,
                tail_atten=args.tail_atten,
            )
            y2 = _norm_peak_match(y2, y)
            y16 = _to_int16_write(y2)
            out_path = batch_dir / wav_path.name
            sf.write(out_path, y16, sr, subtype="PCM_16")
            meta["files"].append(str(out_path.relative_to(BASE)))
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(wavs)}")
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(wavs)} files to {batch_dir}")
        return

    out_root = (BASE / args.out).resolve()
    orig_dir = out_root / "originals"
    proc_dir = out_root / "processed"
    orig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    scores = [_score_clip(w) for w in wavs]
    pick = _pick_examples(scores, max(1, args.count))

    manifest: dict[str, Any] = {
        "disclaimer": (
            "Heuristic scores only. Cannot prove mic distance, headphone bleed, latency, "
            "or specific plugins. Processed files are conservative guesses; listen before batch use."
        ),
        "params": {
            "highpass_hz": args.highpass_hz,
            "sub_alpha": args.sub_alpha,
            "tail_atten": args.tail_atten,
        },
        "picked": pick,
    }

    for row in pick:
        rel = Path(row["path"])
        src = BASE / rel
        name = src.name
        y, sr = sf.read(src, always_2d=False)
        if getattr(y, "ndim", 1) == 2:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float64)
        sf.write(orig_dir / name, y, sr, subtype="PCM_16")

        y2 = _process_one(
            y,
            sr,
            highpass_hz=args.highpass_hz,
            sub_alpha=args.sub_alpha,
            tail_atten=args.tail_atten,
        )
        y2 = _norm_peak_match(y2, y)
        y16 = _to_int16_write(y2)
        sf.write(proc_dir / name, y16, sr, subtype="PCM_16")
        row["out_original"] = str((orig_dir / name).relative_to(BASE))
        row["out_processed"] = str((proc_dir / name).relative_to(BASE))

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(pick)} pairs under {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
