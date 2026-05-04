"""
Track C Step 1 — Synthesize Q&A questions as audio using Piper JOY voice.

Input:  data/qa_pairs.jsonl
Output: data/qa_audio/qa_XXXXXX.wav  (16 kHz mono, resampled from Piper 22050 Hz)
        data/qa_audio/manifest.jsonl  (wav_path | question | answer | category)

Resume-safe: skips WAVs that already exist.

Env vars:
  PIPER_EXE         — path to piper.exe (default: GEMMA4KIDS_DIR/piper/piper.exe)
  PIPER_VOICE_ONNX  — path to .onnx voice model
                      Default: output/piper_voice/el_GR-joy-medium.onnx
                      Auto-fallback to rapunzelina if JOY not yet trained.

Run:
  python training/synthesize_qa_audio.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
QA_PAIRS = DATA_DIR / "qa_pairs.jsonl"
OUT_DIR = DATA_DIR / "qa_audio"
MANIFEST = OUT_DIR / "manifest.jsonl"

PIPER_SAMPLE_RATE = 22050
STT_SAMPLE_RATE = 16000


def _resolve_piper_exe() -> Path:
    env = os.getenv("PIPER_EXE", "").strip()
    if env:
        return Path(env)
    gemma4kids = os.getenv("GEMMA4KIDS_DIR", r"C:\Users\efso office\Desktop\Gemma4Kids").strip()
    return Path(gemma4kids) / "piper" / "piper.exe"


def _resolve_voice_onnx() -> Path:
    env = os.getenv("PIPER_VOICE_ONNX", "").strip()
    if env:
        return Path(env)
    joy = BASE / "output" / "piper_voice" / "el_GR-joy-medium.onnx"
    if joy.exists():
        return joy
    rapunzelina = Path(
        os.getenv("GEMMA4KIDS_DIR", r"C:\Users\efso office\Desktop\Gemma4Kids")
    ) / "voices" / "el_GR-rapunzelina-medium.onnx"
    return rapunzelina


def load_pairs() -> list[dict]:
    if not QA_PAIRS.exists():
        print(f"[ERROR] {QA_PAIRS} not found. Run step D (generate_qa_pipeline.py) first.")
        sys.exit(1)
    pairs = []
    with open(QA_PAIRS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            cat = (obj.get("category") or "general").strip()
            if q and a:
                pairs.append({"question": q, "answer": a, "category": cat})
    return pairs


def synthesize_one(text: str, piper_exe: Path, voice_onnx: Path, out_wav: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [str(piper_exe), "--model", str(voice_onnx), "--output_file", str(tmp_path)],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False

        data, sr = librosa.load(str(tmp_path), sr=None, mono=True)
        if sr != STT_SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=STT_SAMPLE_RATE)
        sf.write(str(out_wav), data, STT_SAMPLE_RATE)
        return True
    except Exception as exc:
        print(f"\n  [WARN] Synthesis failed for text snippet: {exc}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def load_existing_manifest() -> set[str]:
    done: set[str] = set()
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj.get("wav_path", ""))
                except json.JSONDecodeError:
                    pass
    return done


def main() -> None:
    print("=" * 60)
    print("  Gemma4GR Track C — Synthesize Q&A Audio")
    print("=" * 60 + "\n")

    piper_exe = _resolve_piper_exe()
    voice_onnx = _resolve_voice_onnx()

    if not piper_exe.exists():
        print(f"[ERROR] Piper executable not found: {piper_exe}")
        print("  Set PIPER_EXE in .env or install Piper to the default location.")
        sys.exit(1)

    if not voice_onnx.exists():
        print(f"[ERROR] Voice model not found: {voice_onnx}")
        print("  Complete Track B (train_piper.py) first, or set PIPER_VOICE_ONNX in .env.")
        sys.exit(1)

    voice_label = "JOY" if "joy" in voice_onnx.name.lower() else voice_onnx.stem
    print(f"  Piper:  {piper_exe}")
    print(f"  Voice:  {voice_onnx}  [{voice_label}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    print(f"  Q&A pairs loaded: {len(pairs)}")

    done_paths = load_existing_manifest()
    print(f"  Already synthesized: {len(done_paths)}\n")

    manifest_fh = open(MANIFEST, "a", encoding="utf-8")

    ok = skipped = failed = 0
    try:
        for i, pair in enumerate(tqdm(pairs, desc="Synthesizing", unit="wav")):
            wav_name = f"qa_{i:06d}.wav"
            wav_path = OUT_DIR / wav_name
            wav_key = str(wav_path)

            if wav_key in done_paths or wav_path.exists():
                skipped += 1
                continue

            success = synthesize_one(pair["question"], piper_exe, voice_onnx, wav_path)
            if success:
                record = {
                    "wav_path": str(wav_path),
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "category": pair["category"],
                }
                manifest_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_fh.flush()
                ok += 1
            else:
                failed += 1
    finally:
        manifest_fh.close()

    print(f"\n  Done: {ok} synthesized | {skipped} skipped | {failed} failed")
    print(f"  WAVs:     {OUT_DIR}")
    print(f"  Manifest: {MANIFEST}")
    if ok + skipped > 0:
        print("\n  Next: python training/prepare_stt_qa_dataset.py")


if __name__ == "__main__":
    main()
