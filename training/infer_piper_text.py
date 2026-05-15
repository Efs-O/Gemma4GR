"""
Synthesize one text prompt with Piper using the trained local voice by default.

Default model:
  output/piper_voice/el_GR-joy-medium.onnx

Examples:
  python training/infer_piper_text.py --text "Καλημέρα, τι κάνεις;"
  python training/infer_piper_text.py --text "Δοκιμή φωνής." --out output/demo.wav
  python training/infer_piper_text.py --text "Δοκιμή." --model output/piper_voice/el_gr_ly_medium.onnx
  python training/infer_piper_text.py --text "Δοκιμή." --noise-scale 0.7 --length-scale 1.1 --sentence-silence 0.15
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf

try:
    from training.console_encoding import ensure_utf8_console
    from training.synthesize_qa_audio import _resolve_piper_exe, _resolve_voice_onnx
except ImportError:
    from console_encoding import ensure_utf8_console
    from synthesize_qa_audio import _resolve_piper_exe, _resolve_voice_onnx

ensure_utf8_console()
load_dotenv()

BASE = Path(__file__).parent.parent


def _default_out_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BASE / "output" / "piper_infer" / f"infer_{stamp}.wav"


def _resolve_model(model_arg: str) -> Path:
    if model_arg.strip():
        candidate = Path(model_arg)
        if not candidate.is_absolute():
            candidate = (BASE / candidate).resolve()
        return candidate
    return _resolve_voice_onnx()


def _play_wav(path: Path) -> None:
    data, sr = sf.read(str(path), always_2d=False)
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1)
    sd.play(data, sr)
    sd.wait()


def synthesize_text(
    text: str,
    voice_onnx: Path,
    out_wav: Path,
    *,
    noise_scale: float | None = None,
    length_scale: float | None = None,
    noise_w: float | None = None,
    sentence_silence: float | None = None,
    extra_args: str = "",
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    piper_exe = _resolve_piper_exe()
    cmd = [str(piper_exe), "--model", str(voice_onnx), "--output_file", str(out_wav)]
    if noise_scale is not None:
        cmd.extend(["--noise_scale", str(noise_scale)])
    if length_scale is not None:
        cmd.extend(["--length_scale", str(length_scale)])
    if noise_w is not None:
        cmd.extend(["--noise_w", str(noise_w)])
    if sentence_silence is not None:
        cmd.extend(["--sentence_silence", str(sentence_silence)])
    if extra_args.strip():
        cmd.extend(extra_args.split())
    return subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize one text prompt with Piper.")
    parser.add_argument("--text", type=str, default="", help="Text to synthesize")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional ONNX model path. Defaults to output/piper_voice/el_GR-joy-medium.onnx",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output WAV path. Default: output/piper_infer/infer_YYYYMMDD_HHMMSS.wav",
    )
    parser.add_argument("--noise-scale", type=float, default=None, help="Optional Piper noise_scale")
    parser.add_argument("--length-scale", type=float, default=None, help="Optional Piper length_scale")
    parser.add_argument("--noise-w", type=float, default=None, help="Optional Piper noise_w")
    parser.add_argument("--sentence-silence", type=float, default=None, help="Optional Piper sentence_silence")
    parser.add_argument("--extra-args", type=str, default="", help="Extra Piper CLI args")
    parser.add_argument("--play", action="store_true", help="Play the WAV after synthesis")
    args = parser.parse_args()

    text = args.text.strip()
    if not text:
        print("[ERROR] Provide text with --text")
        sys.exit(1)

    piper_exe = _resolve_piper_exe()
    voice_onnx = _resolve_model(args.model)
    out_wav = Path(args.out) if args.out.strip() else _default_out_path()
    if not out_wav.is_absolute():
        out_wav = (BASE / out_wav).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if not piper_exe.exists():
        print(f"[ERROR] Piper executable not found: {piper_exe}")
        sys.exit(1)
    if not voice_onnx.exists():
        print(f"[ERROR] Voice ONNX not found: {voice_onnx}")
        sys.exit(1)

    result = synthesize_text(
        text,
        voice_onnx,
        out_wav,
        noise_scale=args.noise_scale,
        length_scale=args.length_scale,
        noise_w=args.noise_w,
        sentence_silence=args.sentence_silence,
        extra_args=args.extra_args,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.decode("utf-8", errors="replace"))
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace"))
        print("[ERROR] Piper inference failed.")
        sys.exit(result.returncode)

    print(f"Model: {voice_onnx}")
    print(f"Output: {out_wav}")
    if args.play:
        print("Playing...")
        _play_wav(out_wav)


if __name__ == "__main__":
    main()
