"""
Merge qa_pairs.jsonl + voice_sentences.jsonl into a single
training JSONL using the Gemma chat template format already used by train_qa.jsonl.

Output: data/train_qa_combined.jsonl  (~5 726 rows, shuffled)
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

BASE = Path(__file__).parent.parent

_qa_path_raw   = os.getenv("COMBINED_QA_SOURCE", "data/qa_pairs.jsonl").strip()
QA_PATH        = Path(_qa_path_raw) if Path(_qa_path_raw).is_absolute() else BASE / _qa_path_raw
SENT_PATH      = BASE / "data" / "voice_sentences.jsonl"
OUT_PATH       = BASE / "data" / "train_qa_combined.jsonl"

SEED = 42

SYSTEM_PROMPT = (
    "Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. "
    "Μην χρησιμοποιείς Αγγλικά, Ρωσικά ή άλλη γλώσσα, εκτός αν το ζητά "
    "ρητά η ερώτηση. Δώσε άμεση, ουσιαστική και πλήρη απάντηση."
)

# One natural Greek prompt per sentence category
CATEGORY_PROMPTS: dict[str, str] = {
    "talking_to_children":    "Πες κάτι τρυφερό ή ενθαρρυντικό σε ένα παιδί.",
    "children_storytelling":  "Πες μια φράση από παραμύθι ή παιδική ιστορία.",
    "everyday_conversation":  "Πες μια φυσική φράση για καθημερινή συνομιλία.",
    "family_home":            "Πες κάτι που ακούγεται συχνά στο σπίτι ή στην οικογένεια.",
    "school_learning":        "Πες μια φράση που ταιριάζει σε σχολικό περιβάλλον.",
    "polite_service":         "Πες μια ευγενική φράση για εξυπηρέτηση πελάτη.",
    "travel_directions":      "Πες κάτι σχετικό με ταξίδι ή οδηγίες κατεύθυνσης.",
    "healthcare_wellbeing":   "Πες μια φράση για υγεία ή καθημερινή φροντίδα.",
    "work_formal":            "Πες μια επίσημη επαγγελματική φράση.",
    "news_information":       "Πες μια πρόταση με νέα ή γενική πληροφορία.",
    "numbers_dates_money":    "Πες μια πρόταση που περιέχει αριθμούς, ημερομηνίες ή χρηματικά ποσά.",
    "culture_tradition":      "Πες κάτι για ελληνική παράδοση, έθιμο ή πολιτισμό.",
    "nature_environment":     "Πες μια πρόταση για τη φύση ή το περιβάλλον.",
    "food_cooking":           "Πες κάτι για φαγητό ή μαγειρική.",
    "sports_leisure":         "Πες μια φράση για άθληση ή ελεύθερο χρόνο.",
    "technology":             "Πες κάτι σχετικό με τεχνολογία ή σύγχρονα μέσα.",
    "emotions_feelings":      "Πες μια φράση που εκφράζει συναίσθημα.",
    "greetings_farewells":    "Πες έναν φυσικό χαιρετισμό ή αποχαιρετισμό.",
}
DEFAULT_PROMPT = "Πες μια φυσική ελληνική πρόταση."


def gemma_format(question: str, answer: str) -> str:
    return (
        f"<bos><|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{question}<turn|>\n"
        f"<|turn>model\n{answer}<turn|>\n"
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_qa_rows(path: Path) -> list[str]:
    rows = read_jsonl(path)
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if q and a:
            out.append(json.dumps({"text": gemma_format(q, a)}, ensure_ascii=False))
    return out


def load_sentence_rows(path: Path) -> list[str]:
    rows = read_jsonl(path)
    out = []
    for r in rows:
        text     = (r.get("text") or "").strip()
        category = (r.get("category") or "").strip()
        if not text:
            continue
        prompt = CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT)
        out.append(json.dumps({"text": gemma_format(prompt, text)}, ensure_ascii=False))
    return out


def main() -> None:
    print(f"Reading QA pairs  : {QA_PATH}")
    qa_rows   = load_qa_rows(QA_PATH)
    print(f"  Loaded {len(qa_rows)} rows")

    print(f"Reading sentences : {SENT_PATH}")
    sent_rows = load_sentence_rows(SENT_PATH)
    print(f"  Loaded {len(sent_rows)} rows")

    combined = qa_rows + sent_rows
    random.seed(SEED)
    random.shuffle(combined)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(row + "\n")

    print(f"\nTotal rows        : {len(combined)}")
    print(f"Output            : {OUT_PATH}")


if __name__ == "__main__":
    main()
