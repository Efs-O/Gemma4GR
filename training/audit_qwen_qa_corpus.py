from __future__ import annotations

import argparse
import os
import json
import re
from collections import Counter
from pathlib import Path
from urllib import request


BASE = Path(__file__).parent.parent
AUDIT_DIR = BASE / "data" / "audit_qa_20260513"
SOURCE_ORIGINAL = AUDIT_DIR / "source_qa_pairs_original.jsonl"
SOURCE_WORKING = AUDIT_DIR / "source_qa_pairs_working.jsonl"
CLEANED_OUT = AUDIT_DIR / "source_qa_pairs_cleaned.jsonl"
FLAGGED_OUT = AUDIT_DIR / "flagged_rows.jsonl"
REPORT_OUT = AUDIT_DIR / "findings_report.json"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "qwen3.5:397b-cloud").strip()

EXPECTED_CATEGORIES = {
    "history",
    "culture",
    "geography",
    "language",
    "science",
    "everyday",
    "food",
    "children_education",
    "mythology",
    "religion_orthodox",
}

LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]{2,}\b")
MOJIBAKE_RE = re.compile(r"[ÎÏÐÑÃÂ]")
REPEATED_TOKEN_RE = re.compile(r"(\b\S+\b)(?:\s+\1){3,}")
MARKUP_RE = re.compile(r"\*\*|\*=|=\*")
SPACE_RE = re.compile(r"\s{2,}")

# These are acceptable in context for acronym references or etymology rows.
ALLOWED_LATIN_TOKENS = {
    "CERN",
    "UNESCO",
    "RAF",
    "Corpus",
    "Juris",
    "Civilis",
    "mast-o-khiar",
    "boccale",
    "pomodoro",
    "decanus",
}

# Conservative line-specific repairs for the preserved 2500-row Qwen corpus.
LINE_FIXES: dict[int, dict[str, str]] = {
    613: {
        "answer_replace": (
            "Κάθε νησιωτικό σύμπλεγμα έχει αναπτύξει τις own του ιδιαιτερότητες λόγω της γεωγραφικής απομόνωσης.",
            "Κάθε νησιωτικό σύμπλεγμα έχει αναπτύξει τις δικές του ιδιαιτερότητες λόγω της γεωγραφικής απομόνωσης.",
        )
    },
    733: {
        "answer_replace": (
            "Ο μεγάλος επιστήμονας διατύπωσε την αρχή της άνωσης, παρατηρώντας how το νερό ανυψώνεται όταν βυθίζεται ένα σώμα.",
            "Ο μεγάλος επιστήμονας διατύπωσε την αρχή της άνωσης, παρατηρώντας πώς το νερό ανυψώνεται όταν βυθίζεται ένα σώμα.",
        )
    },
    1014: {
        "answer_replace": (
            "Μέσω ομαδικών παιχνιδιών, μαθαίνουν κανόνες συνεργασίας και fair play.",
            "Μέσω ομαδικών παιχνιδιών, μαθαίνουν κανόνες συνεργασίας και ευγενούς άμιλλας.",
        )
    },
    1051: {
        "answer_replace": (
            "Τα παιχνίδια στην αυλή διδάσκουν στα παιδιά κανόνες συνεργασίας και fair play χωρίς την άμεση παρέμβαση των ενηλίκων.",
            "Τα παιχνίδια στην αυλή διδάσκουν στα παιδιά κανόνες συνεργασίας και ευγενούς άμιλλας χωρίς την άμεση παρέμβαση των ενηλίκων.",
        )
    },
    1156: {
        "answer_replace": (
            "Επίσης, πριν το μεσημεριανό φαγητό, τα παιδιά ευχαριστούν για το **μπ**ουκιά τους με σεβασμό.",
            "Επίσης, πριν το μεσημεριανό φαγητό, τα παιδιά ευχαριστούν για τη μπουκιά τους με σεβασμό.",
        )
    },
    1158: {
        "answer_replace": (
            "Οι εκπαιδευτικοί φέρνουν στην τάξη πραγματικά λαχανικά, όπως μια κόκκινη **ντ**ομάτα ή ένα πράσινο αγγούρι, για να τα μελετήσουν τα παιδιά.",
            "Οι εκπαιδευτικοί φέρνουν στην τάξη πραγματικά λαχανικά, όπως μια κόκκινη ντομάτα ή ένα πράσινο αγγούρι, για να τα μελετήσουν τα παιδιά.",
        )
    },
    1256: {
        "question": "Ποια είναι η σημασία των δημοτικών τραγουδιών για τη διατήρηση της ιστορικής μνήμης;"
    },
    1297: {
        "question": "Τι διακρίνει τη γεωγραφία της Δυτικής Μακεδονίας από άλλες περιοχές;"
    },
    1332: {
        "answer_replace": (
            "Οι χειμώνες είναι συνήθως mild και οι καλοκαιρινοί μήνες χαρακτηρίζονται από έντονη ηλιοφάνεια χωρίς ακραίες θερμοκρασίες.",
            "Οι χειμώνες είναι συνήθως ήπιοι και οι καλοκαιρινοί μήνες χαρακτηρίζονται από έντονη ηλιοφάνεια χωρίς ακραίες θερμοκρασίες.",
        )
    },
    1336: {
        "question": "Ποια γεωγραφικά χαρακτηριστικά διαφοροποιούν τα Δωδεκάνησα από τις Κυκλάδες;"
    },
    1358: {
        "question": "Τι διακρίνει τη γεωγραφία της Ρόδου από τα υπόλοιπα νησιά των Δωδεκανήσων;"
    },
    1361: {
        "question": "Ποια χαρακτηριστικά διαφοροποιούν το κλίμα των Κυκλάδων από αυτό της Ηπείρου;"
    },
    1658: {
        "answer_replace": (
            "Η ατμόσφαιρα γίνεται ιδιαίτερα cozy καθώς η φωτιά ζεσταίνει το δωμάτιο και οι παρέες συζητούν ώρες ατέλειωτες.",
            "Η ατμόσφαιρα γίνεται ιδιαίτερα ζεστή και θαλπωρική, καθώς η φωτιά ζεσταίνει το δωμάτιο και οι παρέες συζητούν ώρες ατέλειωτες.",
        )
    },
    1791: {
        "answer_replace": (
            "Νέες λέξεις όπως «ξεμπλοκάρω» ή «ψάχνω online» γίνονται μέρος της καθημερινής ομιλίας των νέων.",
            "Νέες λέξεις και εκφράσεις που συνδέονται με το διαδίκτυο γίνονται μέρος της καθημερινής ομιλίας των νέων.",
        )
    },
    1902: {
        "question": "Τι διακρίνει τη σκαλόπα της Νάξου από άλλα τυριά;"
    },
    1968: {
        "question": "Ποια είναι η πρωινή διαδικασία υποδοχής στο ελληνικό νηπιαγωγείο;"
    },
    2303: {
        "question": "Τι διακρίνει τον ελληνικό καφέ από άλλους τύπους καφέ που σερβίρονται στα καφενεία;"
    },
    2420: {
        "answer_replace": (
            "Ο μύθος εξηγεί thus την ιερή σημασία της φλόγας για τον πολιτισμό.",
            "Ο μύθος εξηγεί έτσι την ιερή σημασία της φλόγας για τον πολιτισμό.",
        )
    },
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_source_line"] = lineno
            rows.append(obj)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and repair the preserved Qwen QA corpus.")
    parser.add_argument(
        "--repair-with-ollama",
        action="store_true",
        help="Use the configured Ollama teacher to rewrite contaminated rows before fallback fixes.",
    )
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    text = SPACE_RE.sub(" ", text.strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def check_ollama() -> bool:
    try:
        with request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=8) as resp:
            data = json.loads(resp.read())
        print(f"  Ollama available at {OLLAMA_URL} | tags={len(data.get('models', []))}")
        return True
    except Exception as exc:
        print(f"  [WARN] Ollama unavailable: {exc}")
        return False


def extract_first_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def repair_with_teacher(row: dict) -> dict | None:
    issues = detect_issues(row)
    prompt = (
        "Διόρθωσε το παρακάτω ζεύγος ερώτησης-απάντησης ώστε να είναι φυσικό, σωστό και καθαρό στα ελληνικά. "
        "Αφαίρεσε ξένες λέξεις ή artifacts αν δεν είναι απολύτως απαραίτητα. "
        "Αν πρόκειται για γλωσσολογική/ετυμολογική αναφορά, μπορείς να κρατήσεις τα ξενόγλωσσα παραθέματα μόνο μέσα σε εισαγωγικά. "
        "Κράτησε την ίδια κατηγορία και το ίδιο βασικό νόημα. "
        "Επέστρεψε μόνο ένα JSON object με ακριβώς τα πεδία question, answer, category.\n\n"
        f"Κατηγορία: {row['category']}\n"
        f"Εντοπισμένα ζητήματα: {', '.join(issues) if issues else 'γενική βελτίωση'}\n"
        f"Ερώτηση: {row['question']}\n"
        f"Απάντηση: {row['answer']}"
    )
    payload = json.dumps(
        {
            "model": TEACHER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Είσαι επιμελητής ελληνικού εκπαιδευτικού dataset. "
                        "Γράφεις μόνο στα ελληνικά, με φυσική γραμματική και καθαρή σύνταξη."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
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
        with request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        content = (data.get("message") or {}).get("content", "").strip()
        obj = extract_first_json_object(content)
        if not obj:
            return None
        question = normalize_spaces((obj.get("question") or "").strip())
        answer = normalize_spaces((obj.get("answer") or "").strip())
        category = (obj.get("category") or row["category"]).strip() or row["category"]
        if not question or not answer:
            return None
        return {
            "question": question,
            "answer": answer,
            "category": category,
            "_source_line": row["_source_line"],
        }
    except Exception:
        return None


def apply_line_fixes(row: dict, use_teacher_repair: bool) -> tuple[dict, list[str]]:
    lineno = row["_source_line"]
    fix = LINE_FIXES.get(lineno)
    changed: list[str] = []

    if use_teacher_repair and fix:
        repaired = repair_with_teacher(row)
        if repaired is not None:
            return repaired, ["teacher_rewritten"]

    if not fix:
        row["question"] = normalize_spaces(row["question"])
        row["answer"] = normalize_spaces(row["answer"])
        return row, changed

    if "question" in fix:
        row["question"] = fix["question"]
        changed.append("question_rewritten")

    if "answer_replace" in fix:
        old, new = fix["answer_replace"]
        if old in row["answer"]:
            row["answer"] = row["answer"].replace(old, new)
            changed.append("answer_fragment_fixed")

    row["question"] = normalize_spaces(row["question"])
    row["answer"] = normalize_spaces(row["answer"])
    return row, changed


def is_language_etymology_context(row: dict) -> bool:
    question = row["question"].lower()
    answer = row["answer"].lower()
    return (
        row["category"] == "language"
        and (
            "προέλευση" in question
            or "λέξη" in question
            or "γλωσσική" in question
            or "λεξιλογ" in question
            or "ρίζες" in answer
            or "προέρχεται" in answer
        )
    )


def extract_unallowed_latin_tokens(row: dict) -> list[str]:
    tokens = LATIN_TOKEN_RE.findall(f"{row['question']} {row['answer']}")
    if not tokens:
        return []
    unallowed: list[str] = []
    for token in tokens:
        if token in ALLOWED_LATIN_TOKENS:
            continue
        if is_language_etymology_context(row) and token in {"online"}:
            continue
        unallowed.append(token)
    return sorted(set(unallowed))


def detect_issues(row: dict) -> list[str]:
    issues: list[str] = []
    question = row["question"]
    answer = row["answer"]
    combined = f"{question} {answer}"

    if row.get("category") not in EXPECTED_CATEGORIES:
        issues.append("invalid_category")
    if not question or not answer:
        issues.append("empty_field")
    if MOJIBAKE_RE.search(combined):
        issues.append("mojibake")
    if MARKUP_RE.search(combined):
        issues.append("markup_artifact")
    if REPEATED_TOKEN_RE.search(combined):
        issues.append("repeated_token_storm")

    latin_tokens = extract_unallowed_latin_tokens(row)
    if latin_tokens:
        issues.append("latin_tokens:" + ",".join(latin_tokens))

    return issues


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not SOURCE_ORIGINAL.exists():
        raise SystemExit(f"Missing source file: {SOURCE_ORIGINAL}")
    if not SOURCE_WORKING.exists():
        raise SystemExit(f"Missing working file: {SOURCE_WORKING}")

    original_rows = read_jsonl(SOURCE_ORIGINAL)
    working_rows = read_jsonl(SOURCE_WORKING)
    if len(original_rows) != len(working_rows):
        raise SystemExit("Original and working files differ in row count; refusing to continue.")

    cleaned_rows: list[dict] = []
    flagged_rows: list[dict] = []
    repair_counter = Counter()
    issue_counter = Counter()
    use_teacher_repair = args.repair_with_ollama and check_ollama()

    for row in working_rows:
        patched, changes = apply_line_fixes(dict(row), use_teacher_repair)
        for item in changes:
            repair_counter[item] += 1

        issues = detect_issues(patched)
        if issues:
            flagged_rows.append(
                {
                    **patched,
                    "audit_issues": issues,
                }
            )
            for issue in issues:
                issue_counter[issue.split(":")[0]] += 1
        else:
            cleaned_rows.append(
                {
                    "question": patched["question"],
                    "answer": patched["answer"],
                    "category": patched["category"],
                }
            )

    write_jsonl(CLEANED_OUT, cleaned_rows)
    write_jsonl(FLAGGED_OUT, flagged_rows)

    report = {
        "source_original": str(SOURCE_ORIGINAL.relative_to(BASE)),
        "source_working": str(SOURCE_WORKING.relative_to(BASE)),
        "cleaned_output": str(CLEANED_OUT.relative_to(BASE)),
        "flagged_output": str(FLAGGED_OUT.relative_to(BASE)),
        "total_rows": len(working_rows),
        "cleaned_rows": len(cleaned_rows),
        "flagged_rows": len(flagged_rows),
        "repair_counts": dict(repair_counter),
        "issue_counts": dict(issue_counter),
        "line_fixes_applied": sorted(LINE_FIXES.keys()),
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Qwen QA audit complete")
    print(f"  Total rows   : {len(working_rows)}")
    print(f"  Cleaned rows : {len(cleaned_rows)}")
    print(f"  Flagged rows : {len(flagged_rows)}")
    print(f"  Teacher path : {'enabled' if use_teacher_repair else 'disabled'}")
    print(f"  Clean file   : {CLEANED_OUT}")
    print(f"  Flagged file : {FLAGGED_OUT}")
    print(f"  Report       : {REPORT_OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
