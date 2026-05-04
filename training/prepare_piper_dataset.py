"""
Step 5 — Build Piper training dataset (LJSpeech format).
Input:  data/piper_audio/*.wav   (22050 Hz mono)
        data/transcripts/*.txt
Output: data/piper_dataset/
          wavs/*.wav
          metadata.csv           (wav_name|text)
"""
import shutil, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE       = Path(__file__).parent.parent
PIPER_AUDIO= BASE / "data" / "piper_audio"
TXT_DIR    = BASE / "data" / "transcripts"
OUT_DIR    = BASE / "data" / "piper_dataset"
WAVS_DIR   = OUT_DIR / "wavs"


def verify_sample_rate(wav_path: Path) -> int:
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.samplerate


def build():
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    skipped = 0

    wavs = sorted(PIPER_AUDIO.glob("*.wav"))
    if not wavs:
        print(f"[ERROR] No WAV files in {PIPER_AUDIO}")
        print("  Run step 3 (generate_pairs.py) first.")
        return

    print(f"  Processing {len(wavs)} audio files ...")
    for wav in wavs:
        txt_path = TXT_DIR / (wav.stem + ".txt")
        if not txt_path.exists():
            skipped += 1
            continue

        sr = verify_sample_rate(wav)
        if sr != 22050:
            print(f"  [WARN] {wav.name}: sample rate {sr} != 22050 — skipping")
            skipped += 1
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue

        dst = WAVS_DIR / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)

        # Piper single-speaker LJSpeech format: id|text
        stem = wav.stem
        lines.append(f"{stem}|{text}")

    meta_path = OUT_DIR / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Piper dataset ready.")
    print(f"  Entries: {len(lines)} | Skipped: {skipped}")
    print(f"  Dataset: {OUT_DIR}")
    print(f"  WAVs:    {WAVS_DIR}")
    print(f"  Manifest:{meta_path}")
    print(f"\n  Minimum recommended for good voice: 3,000 entries")
    print(f"  Current count: {len(lines)}")
    if lines:
        print("\n  Sample manifest rows:")
        for line in lines[:5]:
            print(f"    {line[:140]}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Gemma4GR — Prepare Piper Dataset")
    print("=" * 55 + "\n")
    build()
