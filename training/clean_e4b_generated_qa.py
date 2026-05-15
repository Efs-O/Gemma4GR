"""
Clean the generated E4B QA file with safe automatic fixes and quarantine
suspicious rows.

Actions:
  - backup the original main file
  - strip trivial markup artifacts from answers
  - normalize whitespace / punctuation
  - apply a few safe lexical replacements
  - remove empty answers from the main file
  - move suspicious rows (mixed-script / obvious artifacts / too short) to a flagged file
  - overwrite the main file with the cleaned rows only
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "e4b_self_refine"
MAIN_PATH = DATA_DIR / "qa_pairs_e4b_generated.jsonl"
BACKUP_PATH = DATA_DIR / "qa_pairs_e4b_generated.pre_clean.jsonl"
FLAGGED_PATH = DATA_DIR / "qa_pairs_e4b_generated.flagged.jsonl"
EMPTY_PATH = DATA_DIR / "qa_pairs_e4b_generated.empty.jsonl"

SAFE_REPLACEMENTS = {
    "nectar": "νέκταρ",
    "Nectar": "Νέκταρ",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def greek_latin_cyrillic_profile(text: str) -> tuple[bool, bool, bool]:
    has_greek = False
    has_latin = False
    has_cyrillic = False
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "GREEK" in name:
            has_greek = True
        elif "LATIN" in name:
            has_latin = True
        elif "CYRILLIC" in name:
            has_cyrillic = True
    return has_greek, has_latin, has_cyrillic


def clean_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    for src, dst in SAFE_REPLACEMENTS.items():
        text = text.replace(src, dst)

    text = text.replace("*=", "")
    text = text.replace("=*", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("…", "...")

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def suspicious_reason(answer: str) -> str | None:
    if not answer:
        return "empty"
    if len(answer) < 40:
        return "too_short"
    if any(token in answer for token in ["*=", "=*", "**", "__", "`"]):
        return "markup"

    has_greek, has_latin, has_cyrillic = greek_latin_cyrillic_profile(answer)
    if has_greek and has_cyrillic:
        return "mixed_greek_cyrillic"
    if has_greek and has_latin:
        return "mixed_greek_latin"

    if re.search(r"[{}\[\]<>]", answer):
        return "artifact_chars"

    return None


def main() -> None:
    if not MAIN_PATH.exists():
        raise SystemExit(f"Main file not found: {MAIN_PATH}")

    rows = read_jsonl(MAIN_PATH)
    if not BACKUP_PATH.exists():
        write_jsonl(BACKUP_PATH, rows)

    clean_rows: list[dict] = []
    flagged_rows: list[dict] = []
    empty_rows: list[dict] = []

    for row in rows:
        answer = clean_answer((row.get("answer") or "").strip())
        patched = dict(row)
        patched["answer"] = answer

        reason = suspicious_reason(answer)
        if reason == "empty":
            patched["clean_status"] = "empty"
            empty_rows.append(patched)
            continue
        if reason is not None:
            patched["clean_status"] = "flagged"
            patched["clean_reason"] = reason
            flagged_rows.append(patched)
            continue

        patched["clean_status"] = "clean"
        clean_rows.append(patched)

    write_jsonl(MAIN_PATH, clean_rows)
    write_jsonl(FLAGGED_PATH, flagged_rows)
    write_jsonl(EMPTY_PATH, empty_rows)

    print("=" * 72)
    print("E4B generated QA cleanup complete")
    print(f"Backup        : {BACKUP_PATH}")
    print(f"Main cleaned  : {MAIN_PATH}")
    print(f"Flagged rows  : {FLAGGED_PATH} ({len(flagged_rows)})")
    print(f"Empty rows    : {EMPTY_PATH} ({len(empty_rows)})")
    print(f"Kept rows     : {len(clean_rows)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
