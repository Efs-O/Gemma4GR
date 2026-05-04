"""
Generate a Greek sentence list for human voice recording.

This is separate from the Moira synthetic-audio path:
  - Default: OpenRouter (Gemini 2.5 Flash) generates Greek text prompts; set OPENROUTER_API_KEY in .env.
  - Alternate: VOICE_SENTENCE_PROVIDER=ollama uses a local/cloud Ollama teacher model.
  - A human speaker records the prompts; recordings can be prepared for Piper training.

Design goals:
  - Batched generation to avoid long-context drift
  - Fresh request per batch
  - Optional Ollama unload between batches: `ollama stop <model>`
  - Resume-safe output files

Outputs:
  data/voice_sentences.jsonl
  data/voice_recording_manifest.csv
  data/voice_sentences.txt
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib import request

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOICE_SENTENCE_PROVIDER = os.getenv("VOICE_SENTENCE_PROVIDER", "openrouter").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if VOICE_SENTENCE_PROVIDER == "openrouter":
    MODEL = os.getenv("VOICE_SENTENCE_MODEL", "google/gemini-2.5-flash")
else:
    MODEL = os.getenv("VOICE_SENTENCE_MODEL", os.getenv("TEACHER_MODEL", "qwen3.5:397b-cloud"))
TARGET_COUNT = int(os.getenv("VOICE_SENTENCE_TARGET", "3000"))
BATCH_SIZE = int(os.getenv("VOICE_SENTENCE_BATCH_SIZE", "60"))
TEMPERATURE = float(os.getenv("VOICE_SENTENCE_TEMPERATURE", "0.7"))
MAX_RETRIES = int(os.getenv("VOICE_SENTENCE_MAX_RETRIES", "4"))
KEEP_ALIVE = os.getenv("VOICE_SENTENCE_KEEP_ALIVE", "0s")
UNLOAD_BETWEEN_BATCHES = os.getenv("VOICE_SENTENCE_UNLOAD_MODEL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIN_WORDS = int(os.getenv("VOICE_SENTENCE_MIN_WORDS", "4"))
MAX_WORDS = int(os.getenv("VOICE_SENTENCE_MAX_WORDS", "18"))
MIN_CHARS = int(os.getenv("VOICE_SENTENCE_MIN_CHARS", "18"))
MAX_CHARS = int(os.getenv("VOICE_SENTENCE_MAX_CHARS", "140"))

JSONL_PATH = DATA_DIR / "voice_sentences.jsonl"
TXT_PATH = DATA_DIR / "voice_sentences.txt"
CSV_PATH = DATA_DIR / "voice_recording_manifest.csv"

CATEGORY_SPECS: list[tuple[str, str, float]] = [
    (
        "talking_to_children",
        "Ζεστός, πολύ απλός λόγος για πολύ μικρά παιδιά και προσχολική ηλικία· σύντομες "
        "φράσεις φροντίδας, παιχνιδιού και καθημερινής ρουτίνας, κατάλληλες να τις διαβάζει "
        "φιλική φωνή ενηλίκου δίπλα τους.",
        1.35,
    ),
    ("children_storytelling", "Σύντομες προτάσεις αφηγηματικού ύφους για παραμύθι ή ανάγνωση.", 1.10),
    ("everyday_conversation", "Καθημερινή φυσική ομιλία μεταξύ ενηλίκων.", 1.00),
    ("family_home", "Οικιακές φράσεις, οικογένεια, φροντίδα, ρουτίνα.", 0.95),
    ("school_learning", "Γενικά σχολικά: μάθημα, γράμματα, αριθμοί, ασκήσεις (οποιαδήποτε βαθμίδα).", 1.00),
    (
        "primary_school_teacher_6_12",
        "Δασκάλα σε τάξη δημοτικού (6-12 ετών): οδηγίες, εξηγήσεις, ενθάρρυνση, "
        "κανόνες και ήρεμη διόρθωση, ερωτήσεις κατανόησης, ομαδική εργασία, "
        "αναφορές σε μαθήματα και συμπεριφορά. Ζεστό αλλά όχι «νηπιαγωγείο»· "
        "σεβαστικό, σαφές, προφορικό λεξιλόγιο κατάλληλο για παιδιά που διαβάζουν ήδη.",
        1.18,
    ),
    (
        "primary_school_child_6_12",
        "Προτάσεις σαν να τις λέει παιδί δημοτικού (6-12 ετών): σχολείο, διάλειμμα, φίλοι, "
        "άθλημα/χόμπι, οικογένεια, σκέψεις και μικρά παράπονα με παιδική αλλά όχι βρεφική "
        "γλώσσα. Φυσικό προφορικό ελληνικό· χωρίς αργκό ενηλίκων ή υβριστικά.",
        1.15,
    ),
    ("polite_service", "Ευγενικές φράσεις σε κατάστημα, καφέ, φαρμακείο, ταμείο.", 0.90),
    ("travel_directions", "Μετακινήσεις, σταθμοί, δρόμοι, διαδρομές, ραντεβού.", 0.90),
    ("healthcare_wellbeing", "Ήπιες ιατρικές και καθημερινές φράσεις υγείας.", 0.85),
    ("work_formal", "Επαγγελματικός και ουδέτερος λόγος γραφείου.", 0.80),
    ("news_information", "Σαφές ενημερωτικό ύφος χωρίς υπερβολικά ειδικό λεξιλόγιο.", 0.75),
    ("numbers_dates_money", "Τιμές, ώρες, ημερομηνίες, ποσότητες, όλα γραμμένα ολογράφως.", 0.95),
    ("culture_tradition", "Ελληνική καθημερινότητα, γιορτές, φαγητό, τοπικές αναφορές.", 0.75),
]

SYSTEM_PROMPT = """Είσαι επιμελητής ηχογράφησης για δημιουργία ελληνικού dataset φωνής.

Στόχος:
Να δημιουργήσεις φυσικές, ευανάγνωστες ελληνικές προτάσεις που θα τις διαβάσει μια γυναίκα σε μικρόφωνο.

Κανόνες:
1. Γράφε μόνο στα ελληνικά.
2. Κάθε στοιχείο πρέπει να περιέχει ακριβώς μία φυσική πρόταση.
3. Οι προτάσεις πρέπει να ακούγονται φυσικές όταν διαβάζονται δυνατά.
4. Απόφυγε αγγλικές λέξεις, emoji, hashtags, παρενθέσεις, λίστες και τεχνικό θόρυβο.
5. Γράφε αριθμούς, ώρες, ημερομηνίες και ποσά ολογράφως.
6. Μην επαναλαμβάνεις την ίδια αρχή πρότασης ή την ίδια ιδέα.
7. Απόφυγε πολύ μεγάλες ή πολύ μπερδεμένες προτάσεις.
8. Θέλουμε ποικιλία σε ύφος, ρυθμό και σύνταξη.
9. Μη βάζεις εισαγωγικά γύρω από την πρόταση.
10. Επέστρεψε μόνο έγκυρο JSON array.

Σχήμα εξόδου:
[
  {"text": "Η πρόταση εδώ.", "category": "το_όνομα_της_κατηγορίας"}
]
"""


def check_ollama() -> bool:
    try:
        req = request.Request(f"{OLLAMA_URL}/api/tags")
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            print("  [WARN] Ollama is running but no models are installed.")
            return False
        print(f"  Ollama OK. Available models: {', '.join(models[:5])}")
        if MODEL.endswith("-cloud"):
            print(
                f"  [INFO] Teacher '{MODEL}' is an Ollama cloud model"
                " (may not appear in local tags)."
            )
            return True
        if MODEL in models or any(MODEL.split(":")[0] == m.split(":")[0] for m in models):
            return True
        print(f"  [WARN] Exact model '{MODEL}' is not listed by Ollama tags.")
        print(f"  Available: {models}")
        print(f"  To pull: ollama pull {MODEL}")
        return False
    except Exception as exc:
        print(f"  [ERROR] Cannot reach Ollama at {OLLAMA_URL}: {exc}")
        return False


def check_voice_provider() -> bool:
    if VOICE_SENTENCE_PROVIDER == "ollama":
        return check_ollama()
    if VOICE_SENTENCE_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            print("  [ERROR] OPENROUTER_API_KEY is missing in .env")
            return False
        print(f"  OpenRouter OK (model: {MODEL})")
        return True
    print(f"  [ERROR] Unsupported VOICE_SENTENCE_PROVIDER: {VOICE_SENTENCE_PROVIDER}")
    print("  Use 'openrouter' or 'ollama'.")
    return False


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    text = re.sub(r"^\s*\d+[\).\-\s]+", "", text)
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def dedupe_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[«»\"'.,;:!?;…\-–—]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_probably_greek(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    greek = sum(1 for ch in letters if "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff")
    return greek / len(letters) >= 0.80


def word_count(text: str) -> int:
    return len([part for part in re.split(r"\s+", text.strip()) if part])


def sentence_end_count(text: str) -> int:
    return len(re.findall(r"[.!;;?]", text))


def validate_sentence(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    if word_count(text) < MIN_WORDS or word_count(text) > MAX_WORDS:
        return False
    if not is_probably_greek(text):
        return False
    if "\n" in text or "\t" in text:
        return False
    if any(token in text for token in ("http://", "https://", "@", "#")):
        return False
    if sentence_end_count(text) > 2:
        return False
    if text.count(":") > 1:
        return False
    return True


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_array(text: str) -> list[dict]:
    cleaned = strip_code_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("Model did not return a JSON array")
    return [item for item in data if isinstance(item, dict)]


def ollama_generate(prompt: str) -> str | None:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": 8192,
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
        with request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read())
        message = data.get("message", {})
        content = (message.get("content") or "").strip()
        if content:
            return content
        thinking = (message.get("thinking") or "").strip()
        if thinking:
            return thinking
        return None
    except Exception as exc:
        print(f"    [WARN] Ollama request failed: {exc}")
        return None


def openrouter_generate(prompt: str) -> str | None:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": 8192,
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
        with request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"    [WARN] OpenRouter request failed: {exc}")
        return None


def llm_generate(prompt: str) -> str | None:
    if VOICE_SENTENCE_PROVIDER == "openrouter":
        return openrouter_generate(prompt)
    return ollama_generate(prompt)


def maybe_unload_model() -> None:
    if VOICE_SENTENCE_PROVIDER != "ollama":
        return
    if not UNLOAD_BETWEEN_BATCHES:
        return
    try:
        subprocess.run(
            ["ollama", "stop", MODEL],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def load_existing_records() -> list[dict]:
    if not JSONL_PATH.exists():
        return []
    rows: list[dict] = []
    with open(JSONL_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("text") and obj.get("category"):
                rows.append(obj)
    return rows


def category_targets(total: int) -> dict[str, int]:
    weights = {name: weight for name, _, weight in CATEGORY_SPECS}
    weight_sum = sum(weights.values())
    raw = {name: (weights[name] / weight_sum) * total for name in weights}
    base = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(base.values())
    ranked = sorted(weights, key=lambda name: raw[name] - base[name], reverse=True)
    for name in ranked[:remainder]:
        base[name] += 1
    return base


def prompt_for_batch(
    category: str,
    description: str,
    needed: int,
    banned_texts: Iterable[str],
) -> str:
    banned = list(banned_texts)
    avoid_block = ""
    if banned:
        avoid_lines = "\n".join(f"- {text}" for text in banned[:20])
        avoid_block = (
            "\nΑπέφυγε να επιστρέψεις προτάσεις ίδιες ή πολύ κοντινές με αυτές:\n"
            f"{avoid_lines}\n"
        )

    return f"""Κατηγορία: {category}
Περιγραφή: {description}
Ζητούμενο πλήθος: {needed}

Γράψε {needed} μοναδικές προτάσεις κατάλληλες για ηχογράφηση dataset γυναικείας φωνής.
Οι προτάσεις πρέπει:
- να είναι σαφείς όταν διαβάζονται δυνατά
- να είναι μίας πρότασης η καθεμία
- να έχουν περίπου {MIN_WORDS} έως {MAX_WORDS} λέξεις
- να καλύπτουν διαφορετικά μικροθέματα μέσα στην κατηγορία
- να μην μοιάζουν μεταξύ τους
- να μην έχουν αρίθμηση ή σχόλια
- να έχουν πάντα πεδίο "category" με ακριβώς την τιμή "{category}"
{avoid_block}
Επέστρεψε μόνο JSON array με αντικείμενα της μορφής:
[
  {{"text": "Η πρόταση εδώ.", "category": "{category}"}}
]
"""


def rewrite_outputs(records: list[dict]) -> None:
    records = sorted(records, key=lambda row: row["id"])

    with open(JSONL_PATH, "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(TXT_PATH, "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(f"{row['id']}\t[{row['category']}] {row['text']}\n")

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["utterance_id", "category", "text"])
        for row in records:
            writer.writerow([row["id"], row["category"], row["text"]])


def main() -> None:
    print("=" * 68)
    print("  Gemma4GR - Generate Greek Voice Sentence List")
    print(f"  Provider:         {VOICE_SENTENCE_PROVIDER}")
    print(f"  Model:            {MODEL}")
    if VOICE_SENTENCE_PROVIDER == "ollama":
        print(f"  Ollama URL:       {OLLAMA_URL}")
    print(f"  Target sentences: {TARGET_COUNT}")
    print(f"  Batch size:       {BATCH_SIZE}")
    if VOICE_SENTENCE_PROVIDER == "ollama":
        print(f"  Keep alive:       {KEEP_ALIVE}")
        print(f"  Unload per batch: {UNLOAD_BETWEEN_BATCHES}")
    print(f"  Output JSONL:     {JSONL_PATH}")
    print(f"  Output CSV:       {CSV_PATH}")
    print(f"  Output TXT:       {TXT_PATH}")
    print("=" * 68 + "\n")

    if VOICE_SENTENCE_PROVIDER == "ollama":
        print(f"  Manual fresh-session command if needed: ollama stop {MODEL}\n")

    if not check_voice_provider():
        sys.exit(1)

    existing = load_existing_records()
    key_set = {dedupe_key(row["text"]) for row in existing}
    counts: dict[str, int] = {}
    for row in existing:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    targets = category_targets(TARGET_COUNT)

    print(f"  Already have: {len(existing)} sentences")
    for name, _, _ in CATEGORY_SPECS:
        print(f"    {name:<22} {counts.get(name, 0):>4}/{targets[name]}")
    print()

    if len(existing) >= TARGET_COUNT:
        print("Already at or above target. Nothing to do.")
        rewrite_outputs(existing[:TARGET_COUNT])
        return

    records = existing[:]
    next_index = len(records) + 1

    for category, description, _weight in CATEGORY_SPECS:
        target_for_category = targets[category]
        while counts.get(category, 0) < target_for_category:
            remaining = target_for_category - counts.get(category, 0)
            requested = min(BATCH_SIZE, remaining)

            recent_same_category = [
                row["text"]
                for row in records
                if row["category"] == category
            ][-20:]

            print(
                f"  [{category}] need {remaining} more "
                f"(requesting batch of {requested})"
            )

            accepted: list[dict] = []
            for attempt in range(1, MAX_RETRIES + 1):
                prompt = prompt_for_batch(category, description, requested, recent_same_category)
                raw = llm_generate(prompt)
                if not raw:
                    if attempt < MAX_RETRIES:
                        time.sleep(2)
                    continue

                try:
                    candidates = extract_json_array(raw)
                except Exception as exc:
                    print(f"    [WARN] Could not parse JSON: {exc}")
                    candidates = []

                for item in candidates:
                    text = normalize_text(str(item.get("text", "")))
                    item_category = normalize_text(str(item.get("category", category)))
                    if item_category != category:
                        continue
                    if not validate_sentence(text):
                        continue
                    key = dedupe_key(text)
                    if not key or key in key_set:
                        continue
                    accepted.append({"text": text, "category": category})
                    key_set.add(key)
                    if len(accepted) >= requested:
                        break

                if len(accepted) >= requested:
                    break

                if attempt < MAX_RETRIES:
                    time.sleep(2)

            if not accepted:
                print(f"    [WARN] No valid sentences accepted for {category}; stopping this category.")
                break

            for row in accepted[:requested]:
                row["id"] = f"gr_voice_{next_index:06d}"
                row["model"] = MODEL
                records.append(row)
                counts[category] = counts.get(category, 0) + 1
                next_index += 1

            rewrite_outputs(records)
            print(
                f"    Saved {len(accepted[:requested])} items. "
                f"Category total: {counts.get(category, 0)}/{target_for_category} | "
                f"Overall: {len(records)}/{TARGET_COUNT}"
            )
            maybe_unload_model()
            time.sleep(1)

            if len(records) >= TARGET_COUNT:
                break

        if len(records) >= TARGET_COUNT:
            break

    print("\n" + "=" * 68)
    print(f"  Done. Final sentence count: {len(records)}")
    print(f"  JSONL: {JSONL_PATH}")
    print(f"  CSV:   {CSV_PATH}")
    print(f"  TXT:   {TXT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
