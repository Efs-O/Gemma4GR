"""
Greek Q&A benchmark — call via run_before.py or run_after.py.
Uses llama.cpp server (started from Gemma4Kids path).
"""
import json, os, subprocess, sys, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE         = Path(__file__).parent.parent
RESULTS_DIR  = BASE / "tests" / "benchmark_results"
LLAMA_PORT   = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_URL    = f"http://localhost:{LLAMA_PORT}/v1/chat/completions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Test suite ────────────────────────────────────────────────────────────
QA_SUITE = [
    # Factual Greek
    {"q": "Ποια είναι η πρωτεύουσα της Ελλάδας;",
     "must_contain": ["Αθήνα"], "category": "factual"},
    {"q": "Πόσες μέρες έχει μια εβδομάδα;",
     "must_contain": ["επτά", "7"], "category": "factual"},
    {"q": "Ποιος έγραψε την Ιλιάδα;",
     "must_contain": ["Όμηρο", "Ομηρο"], "category": "factual"},
    {"q": "Ποιο είναι το μεγαλύτερο νησί της Ελλάδας;",
     "must_contain": ["Κρήτη"], "category": "factual"},
    {"q": "Ποιος ήταν ο Σωκράτης;",
     "must_contain": ["φιλόσοφ", "Αθήν"], "category": "factual"},
    # Numbers
    {"q": "Πες μου τον αριθμό 1234 με γράμματα.",
     "must_contain": ["χίλια", "διακόσι"], "category": "numbers"},
    {"q": "Πόσο κάνει πέντε επί έξι;",
     "must_contain": ["τριάντα", "30"], "category": "numbers"},
    {"q": "Ποια χρονολογία είναι εκατό χρόνια μετά το 1900;",
     "must_contain": ["2000", "δύο χιλιάδες"], "category": "numbers"},
    # Grammar
    {"q": "Γράψε μία πρόταση στον παρακείμενο χρόνο με το ρήμα 'τρώω'.",
     "must_contain": ["έχω φάει", "έχεις φάει", "έφαγ"], "category": "grammar"},
    {"q": "Ποια είναι η πληθυντική του 'σπίτι';",
     "must_contain": ["σπίτια"], "category": "grammar"},
    {"q": "Κλίνε το άρθρο 'ο' στη γενική πτώση.",
     "must_contain": ["του"], "category": "grammar"},
    # Translation
    {"q": "Μετέφρασε στα αγγλικά: 'Καλημέρα'",
     "must_contain": ["Good morning"], "category": "translation"},
    {"q": "Μετέφρασε στα ελληνικά: 'Thank you very much'",
     "must_contain": ["Ευχαριστώ", "ευχαριστώ"], "category": "translation"},
    {"q": "Τι σημαίνει 'φιλοξενία' στα αγγλικά;",
     "must_contain": ["hospitality", "welcom"], "category": "translation"},
    # Culture
    {"q": "Τι είναι το μουσακά;",
     "must_contain": ["φαγητ", "μελιτζ", "κιμά"], "category": "culture"},
    {"q": "Πότε γιορτάζεται η Εθνική Επέτειος στην Ελλάδα;",
     "must_contain": ["25", "Μαρτίου", "Μαρτ"], "category": "culture"},
    {"q": "Τι είναι η δημοκρατία;",
     "must_contain": ["λαό", "κυβέρν", "εξουσ"], "category": "culture"},
    # Instruction following
    {"q": "Γράψε μία συνταγή για τσατσίκι.",
     "must_contain": ["γιαούρτ", "αγγούρ", "σκόρδ"], "category": "instruction"},
    {"q": "Δώσε μου τρεις λόγους να επισκεφτώ την Ελλάδα.",
     "must_contain": ["ιστορ", "θάλασσ", "κουζίν", "νησ", "πολιτισμ"], "category": "instruction"},
    # Dialect/slang
    {"q": "Τι σημαίνει 'ρε' στα ελληνικά;",
     "must_contain": None, "category": "dialect"},
    {"q": "Τι σημαίνει 'γεια' στα ελληνικά;",
     "must_contain": ["χαιρετισμ", "hello", "hi", "υγεία"], "category": "dialect"},
]


def ask(question: str, model_label: str) -> str:
    """Send a question to the llama.cpp server and return the answer."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(LLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def run_qa_benchmark(model_label: str) -> dict:
    """Run full Q&A benchmark. Returns result dict."""
    print(f"\n  Running Greek Q&A benchmark [{model_label}] ...")
    print(f"  Questions: {len(QA_SUITE)}")

    results = []
    passed  = 0

    for item in QA_SUITE:
        answer  = ask(item["q"], model_label)
        musts   = item["must_contain"]

        if musts is None:
            ok = True   # open-ended question — always passes
        else:
            ok = any(m.lower() in answer.lower() for m in musts)

        if ok:
            passed += 1

        results.append({
            "category":     item["category"],
            "question":     item["q"],
            "answer":       answer,
            "must_contain": musts,
            "passed":       ok,
        })

        status = "✓" if ok else "✗"
        print(f"    [{status}] {item['q'][:60]}")

    score = passed / len(QA_SUITE) * 100
    out = {
        "model":     model_label,
        "score_pct": round(score, 1),
        "passed":    passed,
        "total":     len(QA_SUITE),
        "results":   results,
    }

    out_path = RESULTS_DIR / f"{model_label}_qa.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Score: {passed}/{len(QA_SUITE)} ({score:.1f}%)")
    print(f"  Saved: {out_path}")
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    run_qa_benchmark(label)
