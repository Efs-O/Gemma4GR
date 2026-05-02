"""
Greek STT benchmark using the llama.cpp chat completions endpoint.
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

BASE = Path(__file__).parent.parent
CLIPS_DIR = BASE / "tests" / "test_clips"
RESULTS_DIR = BASE / "tests" / "benchmark_results"
LLAMA_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_URL = f"http://localhost:{LLAMA_PORT}/v1/chat/completions"
STT_BENCH_LIMIT = int(os.getenv("STT_BENCH_LIMIT", "0"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Transcribe the following speech segment in Greek into Greek text."


def transcribe(wav_path: Path) -> str:
    import base64

    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(LLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[ERROR: {exc}]"


def normalize(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFC", text.lower().strip())
    text = re.sub(r"[.,;:!?«»\"'()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_stt_benchmark(model_label: str) -> dict:
    try:
        import jiwer
    except ImportError:
        print("[ERROR] jiwer not installed. Run: pip install jiwer")
        sys.exit(1)

    clips = sorted(CLIPS_DIR.glob("*.wav"))
    if STT_BENCH_LIMIT > 0:
        clips = clips[:STT_BENCH_LIMIT]

    if not clips:
        print(f"[WARN] No test clips in {CLIPS_DIR}")
        print("  Add 10-20 Greek WAV (16kHz) + matching .txt files to tests/test_clips/")
        return {"model": model_label, "WER": None, "CER": None, "count": 0}

    print(f"\n  Running STT benchmark [{model_label}]")
    print(f"  Clips: {len(clips)}")

    hypotheses = []
    references = []
    details = []

    for wav in clips:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        ref = normalize(txt.read_text(encoding="utf-8"))
        hyp = normalize(transcribe(wav))

        hypotheses.append(hyp)
        references.append(ref)

        wer_clip = jiwer.wer([ref], [hyp]) * 100
        details.append({"file": wav.name, "reference": ref, "hypothesis": hyp, "WER": round(wer_clip, 1)})
        status = "OK" if wer_clip < 30 else "~" if wer_clip < 60 else "FAIL"
        print(f"    [{status}] {wav.name}: WER={wer_clip:.0f}%")

    if not hypotheses:
        print("  [WARN] No matching .txt files found.")
        return {"model": model_label, "WER": None, "CER": None, "count": 0}

    overall_wer = jiwer.wer(references, hypotheses) * 100
    overall_cer = jiwer.cer(references, hypotheses) * 100

    out = {
        "model": model_label,
        "WER": round(overall_wer, 2),
        "CER": round(overall_cer, 2),
        "count": len(hypotheses),
        "details": details,
    }

    out_path = RESULTS_DIR / f"{model_label}_stt.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  WER: {overall_wer:.1f}% | CER: {overall_cer:.1f}%")
    print(f"  Saved: {out_path}")
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    run_stt_benchmark(label)
