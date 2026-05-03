"""
Orpheus-style voice-prefix sweep on Moira: generate N clips per named voice.
Output basenames: {voice}_pair_0000.wav ... (same under raw_audio, resampled, piper, transcripts).
Transcripts contain Greek text only (voice is only in the filename), matching the main pipeline.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import soundfile as sf
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

_spec = importlib.util.spec_from_file_location("_gp", BASE / "training" / "generate_pairs.py")
assert _spec and _spec.loader
_gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gp)

_DEFAULT_VOICES = "tara,leah,jess,leo,dan,mia,zac,zoe"


def _voice_list() -> list[str]:
    raw = os.getenv("ORPHEUS_VOICE_NAMES", _DEFAULT_VOICES)
    return [v.strip().lower() for v in raw.split(",") if v.strip()]


def main() -> None:
    if not _gp.use_moira_lora():
        print("[ERROR] Voice sweep expects full Moira (USE_MOIRA_LORA=1 or unset).")
        sys.exit(1)
    voices = _voice_list()
    n = int(os.getenv("PAIRS_PER_VOICE", "2"))
    if n < 1:
        print("[ERROR] PAIRS_PER_VOICE must be >= 1.")
        sys.exit(1)

    sents = _gp.get_sentences(n)
    print("=" * 55)
    print("  Gemma4GR — Orpheus voice-name sweep (Moira LoRA)")
    print(f"  Voices ({len(voices)}): {', '.join(voices)}")
    print(f"  Clips per voice: {n}")
    print(f"  Orpheus: {_gp.ORPHEUS_DIR}")
    print(f"  Moira adapters: {_gp.moira_adapter_path()}")
    print("=" * 55)

    print("\n  Loading models ...")
    model, tokenizer, snac_model = _gp.load_moira()
    print("  Models loaded.\n")

    failed = 0
    ok = 0
    for voice in voices:
        for i in tqdm(range(n), desc=f"{voice}"):
            greek = sents[i]
            prompt = f"{voice}: {greek}"
            basename = f"{voice}_pair_{i:04d}"
            try:
                audio = _gp.generate_audio(prompt, model, tokenizer, snac_model)
                if audio is None:
                    failed += 1
                    continue
                sf.write(str(_gp.RAW_DIR / f"{basename}.wav"), audio, samplerate=24000)
                (_gp.TXT_DIR / f"{basename}.txt").write_text(greek, encoding="utf-8")
                audio_16k = _gp.resample(audio, 24000, 16000)
                sf.write(str(_gp.STT_DIR / f"{basename}.wav"), audio_16k, samplerate=16000)
                audio_p = _gp.resample(audio, 24000, 22050)
                sf.write(str(_gp.PIPER_DIR / f"{basename}.wav"), audio_p, samplerate=22050)
                ok += 1
            except Exception as e:
                print(f"\n  [WARN] {basename}: {e}")
                failed += 1

    print(f"\n✓ Sweep done. OK: {ok} | Failed: {failed}")
    print(f"  RAW:       {_gp.RAW_DIR}")
    print(f"  STT:       {_gp.STT_DIR}")
    print(f"  Piper:     {_gp.PIPER_DIR}")
    print(f"  Transcripts: {_gp.TXT_DIR}")


if __name__ == "__main__":
    main()
