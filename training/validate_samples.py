"""
Quick sample-first validation gate for generated data.

Checks:
  - STT audio/transcript pairs
  - Piper dataset rows and sample rates
  - Q&A preview/full rows
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

BASE = Path(__file__).parent.parent
SAMPLE_COUNT = int(os.getenv("SAMPLE_COUNT", "5"))
VALIDATION_STATUS = BASE / "logs" / "validation_status.json"


def show_stt_samples() -> None:
    print("\n[STT samples]")
    audio_dir = BASE / "data" / "resampled_audio"
    txt_dir = BASE / "data" / "transcripts"
    wavs = sorted(audio_dir.glob("*.wav"))[:SAMPLE_COUNT]
    if not wavs:
        print("  No STT audio found.")
        return
    for wav in wavs:
        txt = txt_dir / f"{wav.stem}.txt"
        info = sf.info(str(wav))
        transcript = txt.read_text(encoding="utf-8").strip() if txt.exists() else "<missing transcript>"
        print(f"  {wav.name} | {info.samplerate} Hz | {info.frames / info.samplerate:.1f}s | {transcript[:100]}")


def show_piper_samples() -> None:
    print("\n[Piper samples]")
    meta = BASE / "data" / "piper_dataset" / "metadata.csv"
    wav_dir = BASE / "data" / "piper_dataset" / "wavs"
    if not meta.exists():
        print("  No Piper metadata found.")
        return
    rows = [line.strip() for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows[:SAMPLE_COUNT]:
        parts = row.split("|", 1)
        wav = wav_dir / f"{parts[0]}.wav"
        sr = sf.info(str(wav)).samplerate if wav.exists() else "missing"
        text = parts[1] if len(parts) > 1 else "<bad row>"
        print(f"  {parts[0]} | {sr} Hz | {text[:100]}")


def show_qa_samples() -> None:
    print("\n[Q&A samples]")
    preferred = os.getenv("QA_DATASET_SOURCE", "").strip()
    names = ["qa_preview.jsonl", "qa_pairs.jsonl", "qa_pairs_deduped.jsonl"]
    if preferred and preferred not in names:
        names.append(preferred)

    for name in names:
        path = BASE / "data" / name
        if not path.exists():
            continue
        print(f"  Source: {name}")
        with open(path, encoding="utf-8") as f:
            count = 0
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                print(f"    Q: {row.get('question', '')[:100]}")
                print(f"    A: {row.get('answer', '')[:140]}")
                provider = row.get("provider")
                model = row.get("model")
                if provider or model:
                    print(f"    provider={provider} model={model}")
                count += 1
                if count >= SAMPLE_COUNT:
                    break


def write_validation_status() -> None:
    VALIDATION_STATUS.parent.mkdir(parents=True, exist_ok=True)
    tracked = [
        BASE / "data" / "train_stt.jsonl",
        BASE / "data" / "train_stt_val.jsonl",
        BASE / "data" / "piper_dataset" / "metadata.csv",
        BASE / "data" / "qa_preview.jsonl",
        BASE / "data" / "qa_pairs.jsonl",
        BASE / "data" / "qa_pairs_deduped.jsonl",
        BASE / "data" / "train_qa.jsonl",
        BASE / "data" / "val_qa.jsonl",
    ]
    payload = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(path.relative_to(BASE)): path.stat().st_mtime
            for path in tracked
            if path.exists()
        },
    }
    VALIDATION_STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Validation stamp updated: {VALIDATION_STATUS}")


def main() -> None:
    print("=" * 55)
    print("  Gemma4GR - Validate Samples")
    print("=" * 55)
    print("  Run this before long training jobs.\n")
    show_stt_samples()
    show_piper_samples()
    show_qa_samples()
    write_validation_status()


if __name__ == "__main__":
    main()
