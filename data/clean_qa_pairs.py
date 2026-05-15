import json, re, shutil
from pathlib import Path

path = Path(r"c:\Users\efso office\Desktop\Gemma4GR\data\qa_pairs.jsonl")
shutil.copy2(path, path.with_suffix(".jsonl.bak"))
print(f"Backup: {path.with_suffix('.jsonl.bak')}")

lines = path.read_text(encoding="utf-8").splitlines()
pairs = [(i, json.loads(l)) for i, l in enumerate(lines) if l.strip()]

latin_word = re.compile(r'\b[a-zA-Z]{4,}\b')

# Allowlist: legitimate acronyms / proper nouns
ALLOWED = {"CERN", "Juris", "Civilis", "Corpus"}

kept, removed = [], []
for i, p in pairs:
    combined = p["question"] + " " + p["answer"]
    flagged = [w for w in latin_word.findall(combined) if w not in ALLOWED]
    if flagged:
        removed.append((i + 1, flagged[:3], p["question"][:70]))
    else:
        kept.append(json.dumps(p, ensure_ascii=False))

path.write_text("\n".join(kept) + "\n", encoding="utf-8")

print(f"\nRemoved {len(removed)} pairs:")
for lineno, words, q in removed:
    print(f"  line {lineno}: {words} | {q[:60]}")
print(f"\nKept: {len(kept)} pairs")
print(f"New last line: {kept[-1][:80]}")
