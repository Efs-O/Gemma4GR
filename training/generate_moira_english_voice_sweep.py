"""
Orpheus base + Moira LoRA + English lines + `{voice}:` prefix (experimental).

Moira is Greek-trained; English here tests whether named prompts stay usable under LoRA.
Output default: data/moira_english_named/ (override MOIRA_ENGLISH_OUTPUT_DIR).

Uses generate_pairs.load_moira() (Orpheus + Moira LoRA + SNAC).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from env_bootstrap import ensure_unsloth_runtime

ensure_unsloth_runtime(BASE)

_DEFAULT_VOICES = "tara,leah,jess,leo,dan,mia,zac,zoe"

ENGLISH_SENTENCES = [
    "Hello, how are you today?",
    "What are you doing this evening?",
    "Let's grab coffee together.",
    "Where is the nearest subway station?",
    "I'd like to order a cheese pizza.",
    "Can you help me with something?",
    "I don't quite understand what you're saying.",
    "How much does this cost?",
    "Can I pay by card?",
    "Thank you so much for your help.",
    "Please speak a little more slowly.",
    "Where can I find a taxi?",
]


def _voice_list() -> list[str]:
    raw = os.getenv("ORPHEUS_VOICE_NAMES", _DEFAULT_VOICES)
    return [v.strip().lower() for v in raw.split(",") if v.strip()]


def _english_lines(n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        out.extend(ENGLISH_SENTENCES)
    return out[:n]


def main() -> None:
    _spec = importlib.util.spec_from_file_location("_gp", BASE / "training" / "generate_pairs.py")
    assert _spec and _spec.loader
    _gp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gp)
    os.environ["USE_MOIRA_LORA"] = "1"

    rel = os.getenv("MOIRA_ENGLISH_OUTPUT_DIR", "data/moira_english_named").strip()
    root = BASE / rel if not Path(rel).is_absolute() else Path(rel)
    _gp.RAW_DIR = root / "raw_audio"
    _gp.TXT_DIR = root / "transcripts"
    _gp.STT_DIR = root / "resampled_audio"
    _gp.PIPER_DIR = root / "piper_audio"
    for d in (_gp.RAW_DIR, _gp.TXT_DIR, _gp.STT_DIR, _gp.PIPER_DIR):
        d.mkdir(parents=True, exist_ok=True)

    voices = _voice_list()
    n = int(os.getenv("PAIRS_PER_VOICE", "2"))
    if n < 1:
        print("[ERROR] PAIRS_PER_VOICE must be >= 1.")
        sys.exit(1)

    lines = _english_lines(n)

    import soundfile as sf

    print("=" * 55)
    print("  Gemma4GR — Orpheus + Moira LoRA — English + name sweep")
    print(f"  Voices ({len(voices)}): {', '.join(voices)}")
    print(f"  Clips per voice: {n}")
    print(f"  Orpheus: {_gp.ORPHEUS_DIR}")
    print(f"  Moira:   {_gp.moira_adapter_path()}")
    print(f"  Out:     {root}")
    print("=" * 55)

    print("\n  Loading models ...")
    model, tokenizer, snac_model = _gp.load_moira()
    print("  Models loaded.\n")

    failed = 0
    ok = 0
    for voice in voices:
        for i in tqdm(range(n), desc=f"{voice}"):
            english = lines[i]
            prompt = f"{voice}: {english}"
            basename = f"{voice}_pair_{i:04d}"
            try:
                audio = _gp.generate_audio(prompt, model, tokenizer, snac_model)
                if audio is None:
                    failed += 1
                    continue
                sf.write(str(_gp.RAW_DIR / f"{basename}.wav"), audio, samplerate=24000)
                (_gp.TXT_DIR / f"{basename}.txt").write_text(english, encoding="utf-8")
                audio_16k = _gp.resample(audio, 24000, 16000)
                sf.write(str(_gp.STT_DIR / f"{basename}.wav"), audio_16k, samplerate=16000)
                audio_p = _gp.resample(audio, 24000, 22050)
                sf.write(str(_gp.PIPER_DIR / f"{basename}.wav"), audio_p, samplerate=22050)
                ok += 1
            except Exception as e:
                print(f"\n  [WARN] {basename}: {e}")
                failed += 1

    print(f"\nDone. OK: {ok} | Failed: {failed}")
    print(f"  RAW: {_gp.RAW_DIR}")
    print(f"  TXT: {_gp.TXT_DIR}")


if __name__ == "__main__":
    main()
