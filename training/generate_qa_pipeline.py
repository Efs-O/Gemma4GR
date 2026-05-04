"""
Phase 2 — Greek Q&A generation pipeline (Gemini-native, batch mode).

Same architecture as generate_voice_sentence_list.py:
  - LLM generates both questions AND answers per batch (no templates, no ceiling)
  - Batched, resume-safe, deduplicated by normalized question text
  - Target: 2500 unique Q&A pairs across 10 categories

Env vars:
  QA_PROVIDER       gemini (default), openrouter, ollama
  QA_MODE           full (default) or preview
  NUM_QA            2500
  QA_BATCH_SIZE     6  (small batches keep OpenRouter output tokens per request low)
  QA_TEMPERATURE    0.80
  QA_MIN_ANSWER_CHARS  minimum answer length to accept (default 90; was 120)
  QA_GEMINI_TIMEOUT HTTP read timeout seconds for native Gemini (default 240)
  QA_OPENROUTER_TIMEOUT  same for OpenRouter (default 240)
  QA_OPENROUTER_MAX_TOKENS  OpenRouter completion cap (default 3072)
  QA_GEMINI_MAX_OUTPUT  native Gemini maxOutputTokens (default 4096)
  GEMINI_API_KEY    required for gemini
  OPENROUTER_API_KEY  required for openrouter
  GEMINI_MODEL      gemini-2.5-flash (default)
  OPENROUTER_MODEL  google/gemini-2.5-flash (default)
  TEACHER_MODEL     qwen3.5:397b-cloud (ollama default)
  GEMINI_RPM_DELAY  6.5  (seconds between Gemini calls, free-tier safe)
  QA_OUTPUT_FILE    override output path

Run:
  python training/generate_qa_pipeline.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import request

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

QA_PROVIDER = os.getenv("QA_PROVIDER", "gemini").strip().lower()
QA_MODE = os.getenv("QA_MODE", "full").strip().lower()
NUM_QA = int(os.getenv("NUM_QA", "2500"))
QA_BATCH_SIZE = int(os.getenv("QA_BATCH_SIZE", "6"))
QA_PREVIEW_COUNT = int(os.getenv("QA_PREVIEW_COUNT", "15"))
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.80"))
MAX_RETRIES = 4
MIN_ANSWER_CHARS = int(os.getenv("QA_MIN_ANSWER_CHARS", "90"))
GEMINI_RPM_DELAY = float(os.getenv("GEMINI_RPM_DELAY", "6.5"))
GEMINI_HTTP_TIMEOUT = float(os.getenv("QA_GEMINI_TIMEOUT", "240"))
OPENROUTER_HTTP_TIMEOUT = float(os.getenv("QA_OPENROUTER_TIMEOUT", "240"))
OPENROUTER_MAX_TOKENS = int(os.getenv("QA_OPENROUTER_MAX_TOKENS", "3072"))
GEMINI_MAX_OUTPUT = int(os.getenv("QA_GEMINI_MAX_OUTPUT", "4096"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OLLAMA_MODEL = os.getenv("TEACHER_MODEL", "qwen3.5:397b-cloud")

_custom_out = os.getenv("QA_OUTPUT_FILE", "").strip()
OUT_FILE = Path(_custom_out) if _custom_out else DATA_DIR / "qa_pairs.jsonl"

# ── Categories ───────────────────────────────────────────────────────────────

CATEGORY_SPECS: list[tuple[str, str]] = [
    (
        "history",
        "Αρχαία Ελλάδα, Βυζάντιο, Επανάσταση 1821, Β' Παγκόσμιος, σύγχρονη ιστορία — "
        "πρόσωπα, γεγονότα, μάχες, εποχές, αιτίες και συνέπειες ιστορικών στιγμών.",
    ),
    (
        "culture",
        "Παραδόσεις, ήθη, γιορτές, μουσική (ρεμπέτικο, δημοτικά), θέατρο, φιλοξενία, "
        "καρναβάλι, ελληνική ταυτότητα, λαϊκή τέχνη και έθιμα.",
    ),
    (
        "geography",
        "Ελληνικά νησιά, ορεινοί όγκοι, ποτάμια, λίμνες, περιφέρειες, σύνορα, "
        "κλίμα, αξιοθέατα, διαφορές μεταξύ περιοχών.",
    ),
    (
        "language",
        "Ελληνικό αλφάβητο, διαλέκτοι, ετυμολογία, γραμματική, η επιρροή της ελληνικής "
        "στις ευρωπαϊκές γλώσσες, δημοτική vs καθαρεύουσα, σύγχρονες γλωσσικές τάσεις.",
    ),
    (
        "science",
        "Αρχαίοι Έλληνες επιστήμονες (Αρχιμήδης, Ιπποκράτης, Ευκλείδης), ανακαλύψεις, "
        "σύγχρονη ελληνική επιστήμη, μαθηματικά, ιατρική, αστρονομία.",
    ),
    (
        "everyday",
        "Καθημερινή ζωή στην Ελλάδα: χαιρετισμοί, μεταφορές, σχολείο, εργασία, "
        "υγεία, αγορές, ψυχαγωγία, αθλητισμός, κοινωνικές αλληλεπιδράσεις.",
    ),
    (
        "food",
        "Παραδοσιακά ελληνικά πιάτα, συνταγές, υλικά, ποτά, ο ελληνικός καφές, "
        "μεσογειακή διατροφή, περιφερειακές γαστρονομικές ιδιαιτερότητες.",
    ),
    (
        "children_education",
        "Ελληνικό εκπαιδευτικό σύστημα, παιδικά παιχνίδια και παραμύθια, αριθμοί "
        "και αλφάβητο για μικρά παιδιά, σχολικές ρουτίνες, παιδαγωγική.",
    ),
    (
        "mythology",
        "Δώδεκα Θεοί, ήρωες (Ηρακλής, Αχιλλέας, Οδυσσέας), μύθοι δημιουργίας, "
        "τέρατα, μαντεία, ο Κάτω Κόσμος, σύνδεση μύθου με τοπωνύμια.",
    ),
    (
        "religion_orthodox",
        "Ορθόδοξη Εκκλησία, Πάσχα, Χριστούγεννα, ονομαστήρια, βάπτιση, γάμος, "
        "Άγιον Όρος, σημαντικοί άγιοι, ρόλος της εκκλησίας στη σύγχρονη Ελλάδα.",
    ),
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Είσαι ειδικός εκπαιδευτής ελληνικής γλώσσας και πολιτισμού.
Δημιουργείς ζεύγη ερωτήσεων-απαντήσεων στα ελληνικά για εκπαιδευτικό dataset AI.

Κανόνες:
1. Γράφε ΜΟΝΟ στα ελληνικά — μηδέν αγγλικές λέξεις στην έξοδο
2. Κάθε ερώτηση να είναι μοναδική και να καλύπτει διαφορετική πτυχή της κατηγορίας
3. Οι απαντήσεις: 2-4 σύντομες προτάσεις, πλήρεις και σαφείς — όχι μακροσκελή κείμενα ούτε επαναλήψεις
4. Χρησιμοποίησε ποικίλη σύνταξη· κράτα συνολογικά το JSON περιεκτικό (μετρημένο μήκος)
5. Αποφύγε επαναλαμβανόμενες δομές ερωτήσεων («Τι είναι...» μόνο σε μερικές)
6. Επέστρεψε μόνο έγκυρο JSON array — χωρίς κείμενο πριν ή μετά"""


def batch_prompt(category: str, description: str, n: int, recent: list[str]) -> str:
    avoid = ""
    if recent:
        lines = "\n".join(f"- {q}" for q in recent[:15])
        avoid = f"\nΑπόφυγε ερωτήσεις παρόμοιες με αυτές:\n{lines}\n"

    return (
        f"Κατηγορία: {category}\n"
        f"Περιγραφή: {description}\n"
        f"Ζητούμενο πλήθος: {n} ζεύγη\n"
        f"{avoid}\n"
        f"Δημιούργησε {n} μοναδικά ζεύγη ερώτησης-απάντησης (απαντήσεις σύντομες, 2-4 προτάσεις).\n"
        f"Σχήμα εξόδου (μόνο αυτό):\n"
        f'[\n  {{"question": "Ερώτηση εδώ;", "answer": "Πλήρης απάντηση εδώ.", '
        f'"category": "{category}"}}\n]'
    )


# ── Providers ─────────────────────────────────────────────────────────────────

def gemini_generate(prompt: str) -> str | None:
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": GEMINI_MAX_OUTPUT,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode("utf-8")

    req = request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    backoff = 65
    for attempt in range(4):
        try:
            with request.urlopen(req, timeout=GEMINI_HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read())
            parts = data["candidates"][0]["content"]["parts"]
            text = next((p["text"] for p in parts if not p.get("thought")), None)
            return text.strip() if text else None
        except Exception as exc:
            if "429" in str(exc):
                print(f"    [RATE LIMIT] Gemini 429 — waiting {backoff}s …")
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
            else:
                print(f"    [WARN] Gemini error: {exc}")
                return None
    return None


def openrouter_generate(prompt: str) -> str | None:
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }).encode("utf-8")

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
        with request.urlopen(req, timeout=OPENROUTER_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"    [WARN] OpenRouter error: {exc}")
        return None


def ollama_generate(prompt: str) -> str | None:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 8192},
    }).encode("utf-8")

    req = request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        msg = data.get("message", {})
        return (msg.get("content") or msg.get("thinking") or "").strip() or None
    except Exception as exc:
        print(f"    [WARN] Ollama error: {exc}")
        return None


def llm_generate(prompt: str) -> str | None:
    if QA_PROVIDER == "gemini":
        return gemini_generate(prompt)
    if QA_PROVIDER == "openrouter":
        return openrouter_generate(prompt)
    return ollama_generate(prompt)


# ── Validation & parsing ──────────────────────────────────────────────────────

def is_greek(text: str, threshold: float = 0.75) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    greek = sum(1 for c in letters if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")
    return greek / len(letters) >= threshold


def normalize_q(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[;?;.,!«»\"'\-–—]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_pair(q: str, a: str) -> bool:
    q, a = q.strip(), a.strip()
    if not q or not a:
        return False
    if len(q) < 10 or len(q) > 200:
        return False
    if len(a) < MIN_ANSWER_CHARS:
        return False
    if not is_greek(q) or not is_greek(a):
        return False
    return True


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_pairs(raw: str) -> list[dict]:
    cleaned = strip_fences(raw)
    for attempt in (cleaned, cleaned[cleaned.find("["):cleaned.rfind("]") + 1]):
        try:
            data = json.loads(attempt)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


# ── Persistence ───────────────────────────────────────────────────────────────

def load_existing() -> tuple[list[dict], set[str]]:
    if not OUT_FILE.exists():
        return [], set()
    records: list[dict] = []
    keys: set[str] = set()
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj.get("question", "")
                key = normalize_q(q)
                if key and key not in keys:
                    records.append(obj)
                    keys.add(key)
            except json.JSONDecodeError:
                pass
    return records, keys


# ── Provider check ────────────────────────────────────────────────────────────

def check_provider() -> bool:
    if QA_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            print("  [ERROR] GEMINI_API_KEY missing in .env")
            return False
        print(f"  Gemini OK (model: {GEMINI_MODEL})")
        return True
    if QA_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            print("  [ERROR] OPENROUTER_API_KEY missing in .env")
            return False
        print(f"  OpenRouter OK (model: {OPENROUTER_MODEL})")
        return True
    if QA_PROVIDER == "ollama":
        try:
            req = request.Request(f"{OLLAMA_URL}/api/tags")
            with request.urlopen(req, timeout=5) as resp:
                json.loads(resp.read())
            print(f"  Ollama OK (model: {OLLAMA_MODEL})")
            return True
        except Exception as exc:
            print(f"  [ERROR] Cannot reach Ollama: {exc}")
            return False
    print(f"  [ERROR] Unknown QA_PROVIDER: {QA_PROVIDER}")
    return False


# ── Category targets ──────────────────────────────────────────────────────────

def category_targets(total: int) -> dict[str, int]:
    n = len(CATEGORY_SPECS)
    base = total // n
    remainder = total % n
    targets = {cat: base for cat, _ in CATEGORY_SPECS}
    for cat, _ in CATEGORY_SPECS[:remainder]:
        targets[cat] += 1
    return targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    target = QA_PREVIEW_COUNT if QA_MODE == "preview" else NUM_QA

    print("=" * 64)
    print("  Gemma4GR — Generate Greek Q&A Pairs")
    print(f"  Mode:     {QA_MODE}")
    print(f"  Provider: {QA_PROVIDER}")
    print(f"  Target:   {target} pairs")
    print(f"  Batch:    {QA_BATCH_SIZE} pairs/call")
    if QA_PROVIDER == "openrouter":
        print(
            f"  OpenRouter: max_tokens={OPENROUTER_MAX_TOKENS}, "
            f"timeout={OPENROUTER_HTTP_TIMEOUT}s"
        )
    elif QA_PROVIDER == "gemini":
        print(
            f"  Gemini HTTP timeout: {GEMINI_HTTP_TIMEOUT}s, "
            f"maxOutputTokens: {GEMINI_MAX_OUTPUT}"
        )
    print(f"  Output:   {OUT_FILE}")
    print("=" * 64 + "\n")

    if not check_provider():
        sys.exit(1)

    existing, key_set = load_existing()
    counts: dict[str, int] = {}
    for rec in existing:
        cat = rec.get("category", "")
        counts[cat] = counts.get(cat, 0) + 1

    print(f"  Already have: {len(existing)} pairs\n")

    if len(existing) >= target:
        print(f"Already at target ({target}). Nothing to do.")
        return

    targets = category_targets(target)
    records = existing[:]

    with open(OUT_FILE, "a", encoding="utf-8") as out_fh:
        for category, description in CATEGORY_SPECS:
            cat_target = targets[category]
            while counts.get(category, 0) < cat_target:
                remaining = cat_target - counts.get(category, 0)
                batch_n = min(QA_BATCH_SIZE, remaining)
                recent_qs = [
                    r["question"] for r in records if r.get("category") == category
                ][-15:]

                print(f"  [{category}] need {remaining} more (requesting {batch_n})")

                accepted: list[dict] = []
                for attempt in range(1, MAX_RETRIES + 1):
                    raw = llm_generate(batch_prompt(category, description, batch_n, recent_qs))
                    if not raw:
                        time.sleep(3)
                        continue

                    candidates = extract_pairs(raw)
                    for item in candidates:
                        q = item.get("question", "").strip()
                        a = item.get("answer", "").strip()
                        if not validate_pair(q, a):
                            continue
                        key = normalize_q(q)
                        if key in key_set:
                            continue
                        accepted.append({"question": q, "answer": a, "category": category})
                        key_set.add(key)
                        if len(accepted) >= batch_n:
                            break

                    if len(accepted) >= batch_n:
                        break
                    if attempt < MAX_RETRIES:
                        time.sleep(3)

                if not accepted:
                    print(f"    [WARN] No valid pairs for {category} — skipping batch")
                    break

                for rec in accepted:
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    records.append(rec)
                    counts[category] = counts.get(category, 0) + 1

                total_so_far = len(records)
                print(
                    f"    +{len(accepted)} saved. "
                    f"Category: {counts.get(category,0)}/{cat_target} | "
                    f"Total: {total_so_far}/{target}"
                )

                if QA_PROVIDER == "gemini":
                    time.sleep(GEMINI_RPM_DELAY)

                if total_so_far >= target:
                    break

            if len(records) >= target:
                break

    print(f"\n{'=' * 64}")
    print(f"  Done. Total pairs saved: {len(records)}")
    print(f"  Output: {OUT_FILE}")
    if QA_MODE == "preview":
        print("  Review the output, then set QA_MODE=full and rerun for 2500 pairs.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
