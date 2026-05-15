"""
Copy 5 sample WAVs + .txt transcripts to the HF upload dir samples/ folder.
Run once before hf upload.
"""
import shutil
import csv
from pathlib import Path

SRC_WAVS = Path(r"c:\Users\efso office\Desktop\Gemma4GR\data\human_voice_dataset\wavs")
SRC_META = Path(r"c:\Users\efso office\Desktop\Gemma4GR\data\voice_recording_manifest.csv")
DST = Path(r"N:\.cache\huggingface\hub\gemma-4-E4B-it-GR-v2\samples")

# Pick 5 varied samples across different categories
SAMPLE_IDS = [
    "gr_voice_000002",  # children_storytelling
    "gr_voice_000012",  # culture_tradition (Πάσχα)
    "gr_voice_000015",  # talking_to_children
    "gr_voice_000003",  # everyday_conversation
    "gr_voice_000011",  # numbers_dates_money
]

def main():
    DST.mkdir(parents=True, exist_ok=True)

    transcripts = {}
    with open(SRC_META, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            transcripts[row["utterance_id"]] = row["text"]

    for uid in SAMPLE_IDS:
        wav_src = SRC_WAVS / f"{uid}.wav"
        if not wav_src.exists():
            print(f"  [SKIP] WAV not found: {wav_src}")
            continue
        shutil.copy2(wav_src, DST / f"{uid}.wav")
        txt = transcripts.get(uid, "")
        (DST / f"{uid}.txt").write_text(txt, encoding="utf-8")
        print(f"  {uid}.wav  |  {txt}")

    print(f"\nSamples written to: {DST}")

if __name__ == "__main__":
    main()
