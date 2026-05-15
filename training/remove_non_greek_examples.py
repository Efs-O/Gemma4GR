"""
Remove active dataset rows whose user-visible content contains Latin or Cyrillic
letters, then delete matching QA audio WAVs.

This works in place on the active JSONL files:
  data/qa_pairs.jsonl
  data/train_qa.jsonl
  data/val_qa.jsonl
  data/train_qa_combined.jsonl
  data/qa_audio/manifest.jsonl
  data/train_stt_qa.jsonl
  data/val_stt_qa.jsonl
  data/voice_sentences.jsonl

Backups are written next to each file before modification.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
TURN_RE = re.compile(
    r"<\|turn>user\n(?P<user>.*?)<turn\|>\n<\|turn>model\n(?P<model>.*?)<turn\|>",
    re.S,
)

FILES = {
    "qa_pairs": DATA / "qa_pairs.jsonl",
    "train_qa": DATA / "train_qa.jsonl",
    "val_qa": DATA / "val_qa.jsonl",
    "train_qa_combined": DATA / "train_qa_combined.jsonl",
    "qa_manifest": DATA / "qa_audio" / "manifest.jsonl",
    "train_stt_qa": DATA / "train_stt_qa.jsonl",
    "val_stt_qa": DATA / "val_stt_qa.jsonl",
    "voice_sentences": DATA / "voice_sentences.jsonl",
}


def has_non_greek_text(text: str) -> bool:
    return bool(LATIN_RE.search(text) or CYRILLIC_RE.search(text))


def backup_file(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.pre_non_greek_removal.{STAMP}.bak")
    shutil.copy2(path, backup)
    return backup


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_qa_pairs() -> tuple[set[tuple[str, str]], Counter]:
    removed_pairs: set[tuple[str, str]] = set()
    kept_rows: list[dict] = []
    reasons = Counter()

    for _, row in iter_jsonl(FILES["qa_pairs"]):
        question = row.get("question", "")
        answer = row.get("answer", "")
        if has_non_greek_text(question):
            reasons["question_non_greek"] += 1
            removed_pairs.add((question, answer))
            continue
        if has_non_greek_text(answer):
            reasons["answer_non_greek"] += 1
            removed_pairs.add((question, answer))
            continue
        kept_rows.append(row)

    backup_file(FILES["qa_pairs"])
    write_jsonl(FILES["qa_pairs"], kept_rows)
    return removed_pairs, reasons


def filter_chat_jsonl(path: Path, removed_pairs: set[tuple[str, str]] | None = None) -> Counter:
    kept_rows: list[dict] = []
    reasons = Counter()

    for _, row in iter_jsonl(path):
        text = row.get("text", "")
        match = TURN_RE.search(text)
        if not match:
            reasons["unparsed_kept"] += 1
            kept_rows.append(row)
            continue
        user_text = match.group("user")
        model_text = match.group("model")
        if removed_pairs and (user_text, model_text) in removed_pairs:
            reasons["source_pair_removed"] += 1
            continue
        if has_non_greek_text(user_text):
            reasons["user_non_greek"] += 1
            continue
        if has_non_greek_text(model_text):
            reasons["model_non_greek"] += 1
            continue
        kept_rows.append(row)

    backup_file(path)
    write_jsonl(path, kept_rows)
    return reasons


def filter_voice_sentences() -> Counter:
    kept_rows: list[dict] = []
    reasons = Counter()
    path = FILES["voice_sentences"]

    for _, row in iter_jsonl(path):
        text = row.get("text", "")
        if has_non_greek_text(text):
            reasons["text_non_greek"] += 1
            continue
        kept_rows.append(row)

    backup_file(path)
    write_jsonl(path, kept_rows)
    return reasons


def filter_manifest(removed_pairs: set[tuple[str, str]]) -> tuple[set[str], Counter]:
    kept_rows: list[dict] = []
    removed_wavs: set[str] = set()
    reasons = Counter()
    path = FILES["qa_manifest"]

    for _, row in iter_jsonl(path):
        question = row.get("question", "")
        answer = row.get("answer", "")
        wav_path = row.get("wav_path", "")
        if (question, answer) in removed_pairs:
            reasons["source_pair_removed"] += 1
            if wav_path:
                removed_wavs.add(wav_path)
            continue
        if has_non_greek_text(question):
            reasons["question_non_greek"] += 1
            if wav_path:
                removed_wavs.add(wav_path)
            continue
        if has_non_greek_text(answer):
            reasons["answer_non_greek"] += 1
            if wav_path:
                removed_wavs.add(wav_path)
            continue
        kept_rows.append(row)

    backup_file(path)
    write_jsonl(path, kept_rows)
    return removed_wavs, reasons


def filter_stt(path: Path, removed_wavs: set[str]) -> Counter:
    kept_rows: list[dict] = []
    reasons = Counter()

    for _, row in iter_jsonl(path):
        messages = row.get("messages", [])
        if len(messages) != 2:
            reasons["bad_shape_kept"] += 1
            kept_rows.append(row)
            continue
        user = messages[0].get("content", [])
        assistant = messages[1].get("content", [])
        audio_path = ""
        prompt_text = ""
        answer_text = ""
        if isinstance(user, list) and len(user) >= 2:
            audio_path = str(user[0].get("audio", ""))
            prompt_text = str(user[1].get("text", ""))
        if isinstance(assistant, list) and assistant:
            answer_text = str(assistant[0].get("text", ""))

        if audio_path in removed_wavs:
            reasons["removed_wav"] += 1
            continue
        if has_non_greek_text(prompt_text):
            reasons["prompt_non_greek"] += 1
            continue
        if has_non_greek_text(answer_text):
            reasons["answer_non_greek"] += 1
            continue
        kept_rows.append(row)

    backup_file(path)
    write_jsonl(path, kept_rows)
    return reasons


def delete_wavs(removed_wavs: set[str]) -> tuple[int, int]:
    deleted = 0
    missing = 0
    for raw in sorted(removed_wavs):
        wav = Path(raw)
        if not wav.is_absolute():
            wav = (BASE / wav).resolve()
        if wav.exists():
            wav.unlink()
            deleted += 1
        else:
            missing += 1
    return deleted, missing


def main() -> None:
    if not all(path.exists() for path in FILES.values()):
        missing = [str(path) for path in FILES.values() if not path.exists()]
        raise SystemExit(f"Missing required files: {missing}")

    removed_pairs, qa_pairs_reasons = filter_qa_pairs()
    train_qa_reasons = filter_chat_jsonl(FILES["train_qa"], removed_pairs)
    val_qa_reasons = filter_chat_jsonl(FILES["val_qa"], removed_pairs)
    combined_reasons = filter_chat_jsonl(FILES["train_qa_combined"], removed_pairs)
    voice_reasons = filter_voice_sentences()
    removed_wavs, manifest_reasons = filter_manifest(removed_pairs)
    train_stt_reasons = filter_stt(FILES["train_stt_qa"], removed_wavs)
    val_stt_reasons = filter_stt(FILES["val_stt_qa"], removed_wavs)
    wav_deleted, wav_missing = delete_wavs(removed_wavs)

    print("Removed source QA pairs:", sum(qa_pairs_reasons.values()), dict(qa_pairs_reasons))
    print("Removed train_qa rows:", sum(train_qa_reasons.values()), dict(train_qa_reasons))
    print("Removed val_qa rows:", sum(val_qa_reasons.values()), dict(val_qa_reasons))
    print("Removed combined rows:", sum(combined_reasons.values()), dict(combined_reasons))
    print("Removed voice rows:", sum(voice_reasons.values()), dict(voice_reasons))
    print("Removed manifest rows:", sum(manifest_reasons.values()), dict(manifest_reasons))
    print("Removed train_stt rows:", sum(train_stt_reasons.values()), dict(train_stt_reasons))
    print("Removed val_stt rows:", sum(val_stt_reasons.values()), dict(val_stt_reasons))
    print("Removed QA WAVs:", wav_deleted, "| Missing QA WAVs:", wav_missing)
    print("Backups stamp:", STAMP)


if __name__ == "__main__":
    main()
