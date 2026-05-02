"""
Phase 2 — Provider-aware Greek Q&A generation pipeline.

Use preview mode first with local Ollama:
  QA_MODE=preview
  QA_PROVIDER=ollama

If the preview quality is weak, switch the bulk run to:
  QA_MODE=full
  QA_PROVIDER=openrouter
or
  QA_MODE=full
  QA_PROVIDER=gemini
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib import request

from dotenv import load_dotenv

from generate_qa_pairs import CATEGORIES, SYSTEM_PROMPT, generate_followup_question, make_prompt

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

QA_PROVIDER = os.getenv("QA_PROVIDER", "ollama").strip().lower()
QA_MODE = os.getenv("QA_MODE", "full").strip().lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("TEACHER_MODEL", "qwen3.5:397b-cloud")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

NUM_QA = int(os.getenv("NUM_QA", "1500"))
QA_PREVIEW_COUNT = int(os.getenv("QA_PREVIEW_COUNT", "8"))
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.8"))
MAX_RETRIES = 3
QA_CHUNK_SIZE = int(os.getenv("QA_CHUNK_SIZE", "100"))

PREVIEW_FILE = DATA_DIR / "qa_preview.jsonl"
FULL_FILE = DATA_DIR / "qa_pairs.jsonl"
DEFAULT_OUT_FILE = PREVIEW_FILE if QA_MODE == "preview" else FULL_FILE
QA_OUTPUT_FILE = os.getenv("QA_OUTPUT_FILE", "").strip()
OUT_FILE = Path(QA_OUTPUT_FILE) if QA_OUTPUT_FILE else DEFAULT_OUT_FILE


def provider_model_name() -> str:
    if QA_PROVIDER == "openrouter":
        return OPENROUTER_MODEL
    if QA_PROVIDER == "gemini":
        return GEMINI_MODEL
    return OLLAMA_MODEL


def check_ollama() -> bool:
    try:
        req = request.Request(f"{OLLAMA_URL}/api/tags")
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            print("  [WARN] Ollama is running but no models are installed.")
            return False
        print(f"  Ollama OK. Available models: {', '.join(models[:5])}")
        if OLLAMA_MODEL in models:
            return True

        print(f"  [WARN] Exact model '{OLLAMA_MODEL}' is not listed by Ollama tags.")
        print(f"  Available: {models}")
        print(f"  To pull: ollama pull {OLLAMA_MODEL}")
        return False
    except Exception as e:
        print(f"  [ERROR] Cannot reach Ollama at {OLLAMA_URL}: {e}")
        return False


def check_provider() -> bool:
    if QA_PROVIDER == "ollama":
        return check_ollama()
    if QA_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            print("  [ERROR] OPENROUTER_API_KEY is missing in .env")
            return False
        return True
    if QA_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            print("  [ERROR] GEMINI_API_KEY is missing in .env")
            return False
        return True
    print(f"  [ERROR] Unsupported QA_PROVIDER: {QA_PROVIDER}")
    return False


def ollama_generate(prompt: str, system: str) -> str | None:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": 400,
            },
        }
    ).encode("utf-8")

    req = request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        message = data.get("message", {})
        content = (message.get("content") or "").strip()
        if content:
            return content

        thinking = (message.get("thinking") or "").strip()
        if thinking:
            print("    [WARN] Ollama returned thinking-only output; using it as fallback text")
            return thinking
        return None
    except Exception as e:
        print(f"\n    [WARN] Ollama request failed: {e}")
        return None


def openrouter_generate(prompt: str, system: str) -> str | None:
    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
        }
    ).encode("utf-8")

    req = request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"\n    [WARN] OpenRouter request failed: {e}")
        return None


def gemini_generate(prompt: str, system: str) -> str | None:
    payload = json.dumps(
        {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": 400,
            },
        }
    ).encode("utf-8")

    req = request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"\n    [WARN] Gemini request failed: {e}")
        return None


def generate_answer(prompt: str, system: str) -> str | None:
    if QA_PROVIDER == "openrouter":
        return openrouter_generate(prompt, system)
    if QA_PROVIDER == "gemini":
        return gemini_generate(prompt, system)
    return ollama_generate(prompt, system)


def load_done() -> set[str]:
    if not OUT_FILE.exists():
        return set()
    done = set()
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj.get("question", ""))
            except json.JSONDecodeError:
                pass
    return done


def build_question_pool(done_questions: set[str], remaining: int) -> list[tuple[str, str]]:
    all_questions: list[tuple[str, str]] = []
    scheduled_questions = set(done_questions)
    for category, seeds in CATEGORIES.items():
        for question in seeds:
            if question not in scheduled_questions:
                all_questions.append((category, question))
                scheduled_questions.add(question)

    cat_list = list(CATEGORIES.keys())
    while len(all_questions) < remaining:
        category = random.choice(cat_list)
        seed = random.choice(CATEGORIES[category])
        followup = generate_followup_question(category, seed)
        if followup not in scheduled_questions:
            all_questions.append((category, followup))
            scheduled_questions.add(followup)

    random.shuffle(all_questions)
    return all_questions[:remaining]


def main():
    target_count = QA_PREVIEW_COUNT if QA_MODE == "preview" else NUM_QA

    print("=" * 60)
    print("  Gemma4GR Phase 2 — Generate Greek Q&A Pairs")
    print(f"  Mode:     {QA_MODE}")
    print(f"  Provider: {QA_PROVIDER}")
    print(f"  Model:    {provider_model_name()}")
    if QA_PROVIDER == "ollama":
        print(f"  Endpoint: {OLLAMA_URL}")
    print(f"  Output:   {OUT_FILE}")
    print(f"  Target:   {target_count} pairs")
    print("=" * 60 + "\n")

    if not check_provider():
        sys.exit(1)

    done_questions = load_done()
    done_count = len(done_questions)
    print(f"  Already generated: {done_count} pairs")

    if done_count >= target_count:
        print(f"\nAlready have {done_count} pairs (target {target_count}). Done.")
        return

    remaining = target_count - done_count
    chunk_target = min(remaining, QA_CHUNK_SIZE if QA_MODE == "full" else remaining)
    print(f"  Need to generate:  {remaining} more pairs")
    if QA_MODE == "full":
        print(f"  This run will generate at most: {chunk_target} pairs (QA_CHUNK_SIZE={QA_CHUNK_SIZE})\n")
    else:
        print()

    questions_to_do = build_question_pool(done_questions, chunk_target)
    failed = 0
    generated = 0

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for index, (category, question) in enumerate(questions_to_do, 1):
            print(f"  [{index:4d}/{chunk_target}] [{category}] {question[:55]}...")

            answer = None
            for attempt in range(MAX_RETRIES):
                answer = generate_answer(make_prompt(question, category), SYSTEM_PROMPT)
                if answer and len(answer) > 50:
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)

            if not answer or len(answer) < 50:
                print(f"           [SKIP] Empty/short response after {MAX_RETRIES} tries")
                failed += 1
                continue

            record = {
                "question": question,
                "answer": answer,
                "category": category,
                "provider": QA_PROVIDER,
                "model": provider_model_name(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

            if generated % 25 == 0:
                print(f"\n  Progress: {done_count + generated}/{target_count} pairs saved\n")

    total = done_count + generated
    print(f"\n{'=' * 60}")
    print(f"  Done! Generated: {generated} | Failed: {failed} | Total: {total}")
    print(f"  Output: {OUT_FILE}")
    if QA_MODE == "preview":
        print("  Review this preview file before running the full generation pass.")
    elif total < target_count:
        print(f"  Remaining after this chunk: {target_count - total}")
        print("  Re-run the same command to generate the next chunk.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
