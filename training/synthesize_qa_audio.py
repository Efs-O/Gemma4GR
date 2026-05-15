"""
Track C Step 1 — Synthesize Q&A questions as audio using Piper JOY voice.

Input:  data/qa_pairs.jsonl by default, or QA_DATASET_SOURCE
Output: data/qa_audio/qa_XXXXXX.wav by default, or QA_AUDIO_OUT_DIR
        data/qa_audio/manifest.jsonl by default, or QA_AUDIO_MANIFEST

Synthesizes the QUESTION field of each pair.
The 3 240 human voice sentences are handled separately by prepare_stt_qa_dataset.py.

Resume-safe by default: skips WAVs that already exist unless QA_AUDIO_FORCE_REBUILD=1.

Env vars:
  PIPER_EXE         — path to piper.exe (default: GEMMA4KIDS_DIR/piper/piper.exe)
  PIPER_VOICE_ONNX  — path to .onnx voice model
                      Default: output/piper_voice/el_GR-joy-medium.onnx
                      Auto-fallback to rapunzelina if JOY not yet trained.
  QA_DATASET_SOURCE — input JSONL relative to data/ or absolute path
  QA_AUDIO_OUT_DIR  — output WAV directory for synthesized question audio
  QA_AUDIO_MANIFEST — output manifest path
  QA_AUDIO_LIMIT    — synthesize only the first N loaded pairs (for smoke tests)
  QA_AUDIO_FORCE_REBUILD — overwrite existing WAVs instead of skipping them

Run:
  python training/synthesize_qa_audio.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from training.console_encoding import ensure_utf8_console
except ImportError:
    from console_encoding import ensure_utf8_console

ensure_utf8_console()

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
_qa_source_raw = os.getenv("QA_DATASET_SOURCE", "qa_pairs.jsonl").strip()
QA_PAIRS = Path(_qa_source_raw) if Path(_qa_source_raw).is_absolute() else DATA_DIR / _qa_source_raw
_audio_out_raw = os.getenv("QA_AUDIO_OUT_DIR", "").strip()
OUT_DIR = Path(_audio_out_raw) if _audio_out_raw else DATA_DIR / "qa_audio"
if not OUT_DIR.is_absolute():
    OUT_DIR = BASE / OUT_DIR
_manifest_raw = os.getenv("QA_AUDIO_MANIFEST", "").strip()
MANIFEST = Path(_manifest_raw) if _manifest_raw else OUT_DIR / "manifest.jsonl"
if not MANIFEST.is_absolute():
    MANIFEST = BASE / MANIFEST
QA_AUDIO_LIMIT = int(os.getenv("QA_AUDIO_LIMIT", "0"))
QA_AUDIO_FORCE_REBUILD = os.getenv("QA_AUDIO_FORCE_REBUILD", "0").strip().lower() in ("1", "true", "yes")

PIPER_SAMPLE_RATE = 22050
STT_SAMPLE_RATE = 16000

# ── Greek number → words ──────────────────────────────────────────────────────
_EL_UNITS = [
    '', 'ένα', 'δύο', 'τρία', 'τέσσερα', 'πέντε', 'έξι', 'επτά', 'οκτώ', 'εννέα',
    'δέκα', 'έντεκα', 'δώδεκα', 'δεκατρία', 'δεκατέσσερα', 'δεκαπέντε',
    'δεκαέξι', 'δεκαεπτά', 'δεκαοκτώ', 'δεκαεννέα',
]
_EL_TENS = ['', 'δέκα', 'είκοσι', 'τριάντα', 'σαράντα', 'πενήντα',
            'εξήντα', 'εβδομήντα', 'ογδόντα', 'ενενήντα']
_EL_HUNDS = ['', 'εκατό', 'διακόσια', 'τριακόσια', 'τετρακόσια', 'πεντακόσια',
             'εξακόσια', 'επτακόσια', 'οκτακόσια', 'εννιακόσια']


def _int_to_el(n: int) -> str:
    if n == 0:
        return 'μηδέν'
    if n < 0:
        return 'μείον ' + _int_to_el(-n)
    if n < 20:
        return _EL_UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _EL_TENS[t] + (' ' + _EL_UNITS[u] if u else '')
    if n < 1000:
        h, rest = divmod(n, 100)
        base = 'εκατόν' if h == 1 and rest else _EL_HUNDS[h]
        return base + (' ' + _int_to_el(rest) if rest else '')
    if n < 2000:
        rest = n - 1000
        return 'χίλια' + (' ' + _int_to_el(rest) if rest else '')
    if n < 1_000_000:
        th, rest = divmod(n, 1000)
        return _int_to_el(th) + ' χιλιάδες' + (' ' + _int_to_el(rest) if rest else '')
    return str(n)


_NUMBER_RE = re.compile(r'\b(\d+)\b')
_MOJIBAKE_RE = re.compile(r'[ÎÏÐÑÃÂ]')


def normalize_text_el(text: str) -> str:
    """Replace digit sequences with Greek words."""
    def _replace(m: re.Match) -> str:
        return _int_to_el(int(m.group(1)))
    return _NUMBER_RE.sub(_replace, text)
# ─────────────────────────────────────────────────────────────────────────────


def _contains_greek(text: str) -> bool:
    return any("\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff" for ch in text)


def _looks_mojibake(text: str) -> bool:
    return bool(_MOJIBAKE_RE.search(text))


def _looks_broken_tts(text: str, duration_s: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _contains_greek(stripped) and len(stripped) >= 8 and duration_s < 0.7:
        return True
    if len(stripped) >= 40 and duration_s < 1.0:
        return True
    return False


def _resolve_piper_exe() -> Path:
    env = os.getenv("PIPER_EXE", "").strip()
    if env:
        return Path(env)
    gemma4kids = os.getenv("GEMMA4KIDS_DIR", r"C:\Users\efso office\Desktop\Gemma4Kids").strip()
    root = Path(gemma4kids) / "piper"
    for candidate in (
        root / "piper.exe",
        root / "win" / "piper.exe",
        root / "linux" / "piper",
        root / "mac" / "piper",
    ):
        if candidate.exists():
            return candidate
    return root / "piper.exe"


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
    if QA_AUDIO_LIMIT > 0:
        return pairs[:QA_AUDIO_LIMIT]
    return pairs


def validate_pairs(pairs: list[dict]) -> None:
    bad = []
    for idx, pair in enumerate(pairs, start=1):
        if _looks_mojibake(pair["question"]) or _looks_mojibake(pair["answer"]):
            bad.append((idx, pair["question"][:80]))
        if len(bad) >= 5:
            break
    if bad:
        print("[ERROR] Source Q&A dataset contains mojibake-looking text.")
        for idx, snippet in bad:
            print(f"  line {idx}: {snippet}")
        print("  Refusing to synthesize from suspect text.")
        sys.exit(1)


def synthesize_one(text: str, piper_exe: Path, voice_onnx: Path, out_wav: Path) -> bool:
    text = normalize_text_el(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                str(piper_exe),
                "--model", str(voice_onnx),
                "--output_file", str(tmp_path),
                "--noise-scale", "0.667",
                "--length-scale", "0.847",
                "--noise-w", "0.800",
                "--sentence-silence", "0.200",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False

        data, sr = librosa.load(str(tmp_path), sr=None, mono=True)
        duration_s = len(data) / float(sr) if sr and len(data) else 0.0
        if _looks_broken_tts(text, duration_s):
            print("\n  [WARN] Piper produced suspiciously short audio for this text.")
            print(f"         duration={duration_s:.3f}s | chars={len(text.strip())}")
            print(f"         snippet={text[:100]}")
            return False
        if sr != STT_SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=STT_SAMPLE_RATE)
        sf.write(str(out_wav), data, STT_SAMPLE_RATE)
        return True
    except Exception as exc:
        print(f"\n  [WARN] Synthesis failed for text snippet: {exc}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def backup_file(path: Path, tag: str) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.{tag}.{timestamp}{path.suffix}")
    path.replace(backup)
    return backup


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
    validate_pairs(pairs)
    print(f"  Q&A pairs loaded: {len(pairs)}")

    ok = skipped = failed = 0
    manifest_rows: list[dict] = []
    for i, pair in enumerate(tqdm(pairs, desc="Synthesizing", unit="wav")):
        wav_name = f"qa_{i:06d}.wav"
        wav_path = OUT_DIR / wav_name

        if wav_path.exists() and not QA_AUDIO_FORCE_REBUILD:
            skipped += 1
        else:
            success = synthesize_one(pair["question"], piper_exe, voice_onnx, wav_path)
            if success:
                ok += 1
            else:
                failed += 1
                continue

        manifest_rows.append({
            "wav_path": str(wav_path),
            "question": pair["question"],
            "answer": pair["answer"],
            "category": pair["category"],
        })

    backup = backup_file(MANIFEST, "pre_rebuild")
    with open(MANIFEST, "w", encoding="utf-8") as manifest_fh:
        for row in manifest_rows:
            manifest_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  Done: {ok} synthesized | {skipped} skipped | {failed} failed")
    print(f"  WAVs:     {OUT_DIR}")
    print(f"  Manifest: {MANIFEST}")
    if backup:
        print(f"  Backup:   {backup}")
    if ok + skipped > 0:
        print("\n  Next: python training/prepare_stt_qa_dataset.py")


if __name__ == "__main__":
    main()
