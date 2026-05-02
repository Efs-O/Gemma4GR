"""
Phase 2 — Step D: Generate Greek Q&A pairs via a local teacher model.

Teacher: Ollama at localhost:11434 (default model: qwen3.5:397b-cloud).
Override with env var: TEACHER_MODEL=llama3:70b (or any Ollama model).

Output: data/qa_pairs.jsonl
  {"question": "...", "answer": "...", "category": "..."}

Supports resume: skips pairs already written to output file.
Target: NUM_QA pairs (default 1500, configurable via env).
"""
import os, sys, json, time, random
from pathlib import Path
from urllib import request, error as urllib_error
from dotenv import load_dotenv

load_dotenv()

BASE       = Path(__file__).parent.parent
DATA_DIR   = BASE / "data"
OUT_FILE   = DATA_DIR / "qa_pairs.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "qwen3.5:397b-cloud")
NUM_QA        = int(os.getenv("NUM_QA", "1500"))
TEMPERATURE   = float(os.getenv("QA_TEMPERATURE", "0.8"))
MAX_RETRIES   = 3

# ── Q&A topic categories with seeds ─────────────────────────────────────────
CATEGORIES = {
    "history": [
        "Ποιος ήταν ο Μέγας Αλέξανδρος;",
        "Τι ήταν η Αθηναϊκή Δημοκρατία;",
        "Πότε έγινε η Ελληνική Επανάσταση του 1821;",
        "Ποιος έγραψε την Ιλιάδα και την Οδύσσεια;",
        "Τι ήταν η Μάχη του Μαραθώνα;",
        "Ποιος ήταν ο Σωκράτης;",
        "Τι ήταν οι Ολυμπιακοί Αγώνες στην αρχαιότητα;",
        "Πότε χτίστηκε ο Παρθενώνας;",
        "Ποια ήταν η σημασία της Σπάρτης στην αρχαία Ελλάδα;",
        "Τι ήταν η Βυζαντινή Αυτοκρατορία;",
    ],
    "culture": [
        "Ποιες είναι οι κυριότερες ελληνικές παραδόσεις;",
        "Τι είναι το ελληνικό καρναβάλι;",
        "Πώς γιορτάζεται το Πάσχα στην Ελλάδα;",
        "Ποιοι είναι οι διάσημοι Έλληνες συνθέτες;",
        "Τι είναι το ρεμπέτικο;",
        "Ποια είναι τα χαρακτηριστικά της ελληνικής φιλοσοφίας;",
        "Πώς είναι η ελληνική οικογενειακή ζωή;",
        "Τι σημαίνει η φιλοξενία στην ελληνική κουλτούρα;",
        "Ποιες είναι οι πιο γνωστές ελληνικές γιορτές;",
        "Πώς επηρέασε η αρχαία Ελλάδα τον δυτικό πολιτισμό;",
    ],
    "geography": [
        "Ποια είναι η πρωτεύουσα της Ελλάδας;",
        "Πόσα νησιά έχει η Ελλάδα;",
        "Ποιο είναι το ψηλότερο βουνό της Ελλάδας;",
        "Πού βρίσκεται η Κρήτη;",
        "Τι θάλασσες περιβάλλουν την Ελλάδα;",
        "Ποιες είναι οι μεγαλύτερες ελληνικές πόλεις;",
        "Τι είναι τα Δωδεκάνησα;",
        "Πού βρίσκεται η Αθήνα σε σχέση με το Αιγαίο;",
        "Ποιες χώρες συνορεύουν με την Ελλάδα;",
        "Πόσο μεγάλη είναι η Ελλάδα σε έκταση;",
    ],
    "language": [
        "Πότε δημιουργήθηκε το ελληνικό αλφάβητο;",
        "Ποια είναι η διαφορά μεταξύ δημοτικής και καθαρεύουσας;",
        "Από πού προέρχεται η λέξη 'δημοκρατία';",
        "Πόσα γράμματα έχει το ελληνικό αλφάβητο;",
        "Ποιες ευρωπαϊκές γλώσσες επηρέασε η αρχαία ελληνική;",
        "Τι είναι οι διάλεκτοι της ελληνικής γλώσσας;",
        "Πώς γράφεται η λέξη 'ευχαριστώ' στα ελληνικά;",
        "Ποια είναι τα συνηθισμένα ελληνικά ονόματα;",
        "Τι σημαίνει η λέξη 'φιλοσοφία' στα αρχαία ελληνικά;",
        "Πώς σχηματίζεται ο πληθυντικός στα ελληνικά;",
    ],
    "science": [
        "Ποιος Έλληνας μαθηματικός ανακάλυψε το θεώρημα του Πυθαγόρα;",
        "Τι ήταν η συνεισφορά του Αρχιμήδη στην επιστήμη;",
        "Πώς ο Ιπποκράτης επηρέασε την ιατρική;",
        "Ποιος ήταν ο Αριστοτέλης;",
        "Τι ανακάλυψε ο Ευκλείδης;",
        "Ποια ήταν η συνεισφορά του Ηρόδοτου;",
        "Τι είναι η ατομική θεωρία του Δημόκριτου;",
        "Πώς υπολόγισε ο Ερατοσθένης την περίμετρο της Γης;",
        "Τι γνώριζαν οι αρχαίοι Έλληνες για την αστρονομία;",
        "Ποια είναι τα σύγχρονα επιστημονικά επιτεύγματα της Ελλάδας;",
    ],
    "everyday": [
        "Πώς χαιρετούν οι Έλληνες;",
        "Τι τρώνε οι Έλληνες στο πρωινό;",
        "Πώς είναι η καθημερινή ζωή στην Αθήνα;",
        "Τι μέσα μαζικής μεταφοράς υπάρχουν στην Ελλάδα;",
        "Πώς λειτουργεί το σύστημα υγείας στην Ελλάδα;",
        "Τι γλώσσα μαθαίνουν τα παιδιά στο σχολείο;",
        "Πότε είναι τα σχολικά διαλείμματα στην Ελλάδα;",
        "Πώς γιορτάζουν τα γενέθλια στην Ελλάδα;",
        "Τι αθλήματα αγαπούν οι Έλληνες;",
        "Πώς είναι η αγορά εργασίας στην Ελλάδα;",
    ],
    "food": [
        "Τι είναι ο μουσακάς;",
        "Πώς φτιάχνεται η σπανακόπιτα;",
        "Ποια είναι τα παραδοσιακά ελληνικά γλυκά;",
        "Τι είναι η φέτα;",
        "Πώς φτιάχνεται το τζατζίκι;",
        "Ποια είναι τα πιο δημοφιλή ελληνικά πιάτα;",
        "Τι είναι η μεσογειακή διατροφή;",
        "Πώς πίνουν τον καφέ οι Έλληνες;",
        "Τι είναι ο ελληνικός καφές;",
        "Ποια ψάρια τρώνε παραδοσιακά στην Ελλάδα;",
    ],
    "children_education": [
        "Πώς μαθαίνει ένα παιδί να διαβάζει ελληνικά;",
        "Τι παραμύθια αγαπούν τα ελληνόπουλα;",
        "Πώς μετράμε στα ελληνικά από το 1 έως το 10;",
        "Ποια είναι τα χρώματα στα ελληνικά;",
        "Πώς λέμε τα ζώα στα ελληνικά;",
        "Τι μαθήματα έχουν τα παιδιά στο δημοτικό;",
        "Πώς λέμε τις μέρες της εβδομάδας στα ελληνικά;",
        "Ποιοι είναι οι μήνες του χρόνου στα ελληνικά;",
        "Τι παίζουν τα παιδιά στην Ελλάδα;",
        "Πώς λέμε 'καλημέρα' και 'καληνύχτα' στα ελληνικά;",
    ],
}

SYSTEM_PROMPT = """Είσαι ένας ειδικός εκπαιδευτής ελληνικής γλώσσας και πολιτισμού.
Δημιουργείς ερωτήσεις και απαντήσεις στα ελληνικά.

Κανόνες:
1. Γράφε ΜΟΝΟ στα ελληνικά - ούτε μία αγγλική λέξη
2. Χρησιμοποίησε φυσική, σύγχρονη ελληνική γλώσσα
3. Οι απαντήσεις να είναι πλήρεις και εκπαιδευτικές (3-6 προτάσεις)
4. Αποφύγε λανθασμένη γραμματική ή ορθογραφία
5. Το ύφος να είναι φιλικό και κατανοητό"""


def check_ollama() -> bool:
    try:
        req = request.Request(f"{OLLAMA_URL}/api/tags")
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            print(f"  [WARN] Ollama running but no models found.")
            return False
        print(f"  Ollama OK. Available models: {', '.join(models[:5])}")
        if not any(TEACHER_MODEL.split(":")[0] in m for m in models):
            print(f"  [WARN] '{TEACHER_MODEL}' not found. Available: {models}")
            print(f"  To pull: ollama pull {TEACHER_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] Cannot reach Ollama at {OLLAMA_URL}: {e}")
        print("  Make sure Ollama is running (from Gemma4Kids or standalone).")
        return False


def ollama_generate(prompt: str, system: str) -> str | None:
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 400,
        },
    }).encode("utf-8")

    req = request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"].strip()
    except Exception as e:
        print(f"\n    [WARN] Ollama request failed: {e}")
        return None


def make_prompt(question: str, category: str) -> str:
    return (
        f"Κατηγορία: {category}\n\n"
        f"Ερώτηση: {question}\n\n"
        f"Γράψε μια λεπτομερή, σωστή απάντηση στα ελληνικά (3-6 προτάσεις). "
        f"Ξεκίνα απευθείας με την απάντηση χωρίς εισαγωγή."
    )


def generate_followup_question(category: str, existing: str) -> str:
    """Generate a follow-up question based on category to expand the dataset."""
    templates = [
        f"Ποια είναι τα κυριότερα χαρακτηριστικά που σχετίζονται με {category} στην Ελλάδα;",
        f"Τι άλλο είναι σημαντικό να γνωρίζουμε για {category} στην ελληνική παράδοση;",
        f"Πώς επηρεάζει το θέμα '{existing[:30]}...' την ελληνική κοινωνία σήμερα;",
        f"Ποια είναι η ιστορική σημασία που σχετίζεται με {category} στην Ελλάδα;",
        f"Τι διαφορές υπάρχουν μεταξύ παλιάς και σύγχρονης Ελλάδας σχετικά με {category};",
    ]
    return random.choice(templates)


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


def main():
    print("=" * 60)
    print("  Gemma4GR Phase 2 — Generate Greek Q&A Pairs")
    print(f"  Teacher: {TEACHER_MODEL} @ {OLLAMA_URL}")
    print(f"  Target:  {NUM_QA} pairs")
    print("=" * 60 + "\n")

    if not check_ollama():
        sys.exit(1)

    done_questions = load_done()
    done_count = len(done_questions)
    print(f"  Already generated: {done_count} pairs")

    if done_count >= NUM_QA:
        print(f"\n✓ Already have {done_count} pairs (target {NUM_QA}). Done.")
        return

    remaining = NUM_QA - done_count
    print(f"  Need to generate:  {remaining} more pairs\n")

    # Build question pool from seeds, cycling through categories
    all_questions: list[tuple[str, str]] = []
    for cat, seeds in CATEGORIES.items():
        for q in seeds:
            if q not in done_questions:
                all_questions.append((cat, q))

    # If seeds aren't enough, generate follow-up questions
    cat_list = list(CATEGORIES.keys())
    while len(all_questions) < remaining:
        cat = random.choice(cat_list)
        seed = random.choice(CATEGORIES[cat])
        fq = generate_followup_question(cat, seed)
        if fq not in done_questions:
            all_questions.append((cat, fq))

    random.shuffle(all_questions)
    questions_to_do = all_questions[:remaining]

    failed = 0
    generated = 0

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for i, (category, question) in enumerate(questions_to_do, 1):
            print(f"  [{i:4d}/{remaining}] [{category}] {question[:55]}...")

            answer = None
            for attempt in range(MAX_RETRIES):
                answer = ollama_generate(make_prompt(question, category), SYSTEM_PROMPT)
                if answer and len(answer) > 50:
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)

            if not answer or len(answer) < 50:
                print(f"           [SKIP] Empty/short response after {MAX_RETRIES} tries")
                failed += 1
                continue

            record = {"question": question, "answer": answer, "category": category}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

            if generated % 50 == 0:
                print(f"\n  ─── Progress: {done_count + generated}/{NUM_QA} pairs saved ───\n")

    total = done_count + generated
    print(f"\n{'='*60}")
    print(f"  Done! Generated: {generated} | Failed: {failed} | Total: {total}")
    print(f"  Output: {OUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
