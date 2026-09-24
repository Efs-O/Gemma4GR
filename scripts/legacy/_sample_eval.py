import json, random
from pathlib import Path

data_dir = Path("data")
out_dir = data_dir / "eval_curated"
out_dir.mkdir(exist_ok=True)

pairs = [json.loads(l) for l in (data_dir / "qa_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by_cat = {}
for p in pairs:
    by_cat.setdefault(p["category"], []).append(p)

rng = random.Random(7)
selected = []
for cat in sorted(by_cat):
    pool = by_cat[cat].copy()
    rng.shuffle(pool)
    selected.extend(pool[:2])

out_file = out_dir / "text_cases.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Written {len(selected)} cases to {out_file}\n")
for i, item in enumerate(selected, 1):
    cat = item["category"]
    q   = item["question"]
    a   = item["answer"]
    print(f"[{i:02d}] cat={cat}")
    print(f"  Q: {q}")
    print(f"  A: {a}")
    print()
