"""
Greek Q&A benchmark for llama.cpp-compatible chat completions.
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
RESULTS_DIR = BASE / "tests" / "benchmark_results"
LLAMA_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_URL = f"http://localhost:{LLAMA_PORT}/v1/chat/completions"
QA_BENCH_LIMIT = int(os.getenv("QA_BENCH_LIMIT", "0"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QA_SUITE = [
    {"q": "Ποια είναι η πρωτεύουσα της Ελλάδας;", "must_contain": ["Αθήνα"], "category": "factual"},
    {"q": "Πόσες μέρες έχει μια εβδομάδα;", "must_contain": ["επτά", "7"], "category": "factual"},
    {"q": "Ποιος έγραψε την Ιλιάδα;", "must_contain": ["Όμηρ", "Ομηρ"], "category": "factual"},
    {"q": "Ποιο είναι το μεγαλύτερο νησί της Ελλάδας;", "must_contain": ["Κρήτη"], "category": "factual"},
    {"q": "Ποιος ήταν ο Σωκράτης;", "must_contain": ["φιλόσοφ", "Αθήν"], "category": "factual"},
    {"q": "Πες μου τον αριθμό 1234 με γράμματα.", "must_contain": ["χίλια", "διακό"], "category": "numbers"},
    {"q": "Πόσο κάνει πέντε επί έξι;", "must_contain": ["τριάντα", "30"], "category": "numbers"},
    {"q": "Ποια χρονολογία είναι εκατό χρόνια μετά το 1900;", "must_contain": ["2000", "δύο χιλ"], "category": "numbers"},
    {"q": "Γράψε μία πρόταση στον παρακείμενο χρόνο με το ρήμα 'τρώω'.", "must_contain": ["έχω φάει", "έχεις φάει", "έφαγ"], "category": "grammar"},
    {"q": "Ποια είναι η πληθυντική του 'σπίτι';", "must_contain": ["σπίτια"], "category": "grammar"},
    {"q": "Κλίνε το άρθρο 'ο' στη γενική πτώση.", "must_contain": ["του"], "category": "grammar"},
    {"q": "Μετέφρασε στα αγγλικά: 'Καλημέρα'", "must_contain": ["Good morning"], "category": "translation"},
    {"q": "Μετέφρασε στα ελληνικά: 'Thank you very much'", "must_contain": ["Ευχαριστώ", "ευχαριστώ"], "category": "translation"},
    {"q": "Τι σημαίνει 'φιλοξενία' στα αγγλικά;", "must_contain": ["hospitality", "welcom"], "category": "translation"},
    {"q": "Τι είναι το μουσακά;", "must_contain": ["φαγητ", "μελιτζ", "κιμά"], "category": "culture"},
    {"q": "Πότε γιορτάζεται η Εθνική Επέτειος στην Ελλάδα;", "must_contain": ["25", "Μαρτί"], "category": "culture"},
    {"q": "Τι είναι η δημοκρατία;", "must_contain": ["λαό", "κυβέρν", "εξουσ"], "category": "culture"},
    {"q": "Γράψε μία συνταγή για τζατζίκι.", "must_contain": ["γιαούρτ", "αγγούρ", "σκόρδ"], "category": "instruction"},
    {"q": "Δώσε μου τρεις λόγους να επισκεφτώ την Ελλάδα.", "must_contain": ["ιστορ", "θάλασσ", "κουζίν", "νησ", "πολιτισμ"], "category": "instruction"},
    {"q": "Τι σημαίνει 'ρε' στα ελληνικά;", "must_contain": None, "category": "dialect"},
    {"q": "Τι σημαίνει 'γεια' στα ελληνικά;", "must_contain": ["χαιρετισμ", "hello", "hi", "υγεία"], "category": "dialect"},
]


def ask(question: str) -> str:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(LLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[ERROR: {exc}]"


def run_qa_benchmark(model_label: str) -> dict:
    suite = QA_SUITE[:QA_BENCH_LIMIT] if QA_BENCH_LIMIT > 0 else QA_SUITE

    print(f"\n  Running Greek Q&A benchmark [{model_label}] ...")
    print(f"  Questions: {len(suite)}")

    results = []
    passed = 0

    for item in suite:
        answer = ask(item["q"])
        musts = item["must_contain"]

        if musts is None:
            ok = True
        else:
            ok = any(m.lower() in answer.lower() for m in musts)

        if ok:
            passed += 1

        results.append(
            {
                "category": item["category"],
                "question": item["q"],
                "answer": answer,
                "must_contain": musts,
                "passed": ok,
            }
        )

        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {item['q'][:60]}")

    score = passed / len(suite) * 100 if suite else 0.0
    out = {
        "model": model_label,
        "score_pct": round(score, 1),
        "passed": passed,
        "total": len(suite),
        "results": results,
    }

    out_path = RESULTS_DIR / f"{model_label}_qa.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Score: {passed}/{len(suite)} ({score:.1f}%)")
    print(f"  Saved: {out_path}")
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    run_qa_benchmark(label)
