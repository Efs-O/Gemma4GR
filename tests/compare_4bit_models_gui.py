"""
Desktop comparison harness for the 4-bit base and fine-tuned Gemma4GR GGUFs.

What it does:
  - runs the same 5 Greek text questions on 4 models
  - runs the same 5 Greek WAV questions on 4 models
  - uses deterministic-ish generation settings (temperature 0, fixed seed)
  - shows the active model, prompt, reference, and answer on screen
  - synthesizes every answer through Piper JOY and plays it on the speakers
  - writes a JSON summary with automatic quality heuristics

Run:
  python tests/compare_4bit_models_gui.py
"""
from __future__ import annotations

import base64
import json
import os
import queue
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import requests
import soundfile as sf
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from training.console_encoding import ensure_utf8_console
from training.infer_piper_text import _play_wav, synthesize_text
from training.synthesize_qa_audio import _resolve_piper_exe, _resolve_voice_onnx

ensure_utf8_console()
load_dotenv()

def _find_free_port(start: int = 8088, end: int = 8200) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")

def _resolve_llama_port() -> int:
    env_port = int(os.getenv("LLAMA_SERVER_PORT", "0"))
    if env_port > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", env_port))
                return env_port  # env port is free, use it
            except OSError:
                pass  # env port is taken, fall through to scan
    return _find_free_port()

LLAMA_PORT = _resolve_llama_port()
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
LLAMA_HEALTH_URL = f"http://127.0.0.1:{LLAMA_PORT}/health"
LLAMA_SERVER_EXE = os.getenv("LLAMA_SERVER_EXE", "").strip()
DEFAULT_MANIFEST = BASE / "data" / "qa_audio_smoke" / "manifest.jsonl"
RESULTS_ROOT = BASE / "output" / "model_compare_4bit"
SHARED_CHAT_TEMPLATE = BASE / "tests" / "templates" / "gemma4gr_shared_chat_template.jinja"
DEFAULT_STT_METADATA = BASE / "data" / "human_voice_dataset" / "metadata.csv"
GREEK_STT_PROMPT_PATH = BASE / "greek_stt_prompt.txt"
SYSTEM_PROMPT = (
    "Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. "
    "Μην χρησιμοποιείς Αγγλικά, Ρωσικά ή άλλη γλώσσα, εκτός αν το ζητά ρητά "
    "η ερώτηση. Δώσε άμεση, ουσιαστική και πλήρη απάντηση."
)

STT_SYSTEM_PROMPT = ""  # no system prompt during STT training — match exactly at inference
TEXT_PROMPT_INSTRUCTION = (
    "Απάντησε σύντομα, σωστά και μόνο στη γλώσσα της ερώτησης. "
    "Αν η ερώτηση είναι στα ελληνικά, απάντησε μόνο στα ελληνικά."
)
AUDIO_PROMPT_INSTRUCTION = (
    "Άκουσε προσεκτικά την ερώτηση και απάντησε μόνο στη γλώσσα που χρησιμοποιεί ο ομιλητής. "
    "Αν η ερώτηση είναι στα ελληνικά, απάντησε μόνο στα ελληνικά. "
    "Μην κάνεις μεταγραφή. Δώσε μόνο την απάντηση στο νόημα της ερώτησης."
)
REQUEST_TIMEOUT = 180
SERVER_READY_TIMEOUT_S = 120
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CTX_SIZE = 4096
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_AUDIO_MODE = "spoken_qa"

PIPER_NOISE_SCALE = 0.667
PIPER_LENGTH_SCALE = 0.847
PIPER_NOISE_W = 0.8
PIPER_SENTENCE_SILENCE = 0.2

GREEK_STOPWORDS = {
    "και",
    "στο",
    "στη",
    "στην",
    "στον",
    "των",
    "τον",
    "την",
    "του",
    "της",
    "τα",
    "τις",
    "τους",
    "μια",
    "μία",
    "για",
    "που",
    "από",
    "με",
    "να",
    "σε",
    "ως",
    "είναι",
    "ήταν",
    "στος",
    "στις",
    "στοι",
}

QA_PAIRS_PER_CATEGORY = 1
EVAL_RANDOM_SEED = 42
VOICE_CASES_PER_CATEGORY = 1
DEFAULT_QA_PAIRS = BASE / "data" / "qa_pairs.jsonl"
DEFAULT_VOICE_SENTENCES = BASE / "data" / "voice_sentences.jsonl"
VOICE_WAVS_DIR = BASE / "data" / "human_voice_dataset" / "wavs"
CURATED_TEXT_CASES = BASE / "data" / "eval_curated" / "text_cases.jsonl"
CURATED_AUDIO_CASES = BASE / "data" / "eval_curated" / "audio_cases.jsonl"
OLLAMA_JUDGE_URL = "http://localhost:11434/api/chat"
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen3.5:397b-cloud")
JUDGE_TIMEOUT = 120  # qwen3.5 thinking phase consumes ~500 tokens before emitting the verdict
SKIP_JUDGE = os.getenv("SKIP_JUDGE", "0") == "1"  # set to "1" to disable scoring



@dataclass
class ModelSpec:
    key: str
    label: str
    path: str
    mmproj_path: str
    chat_template: str


@dataclass
class CaseResult:
    model_key: str
    model_label: str
    case_id: str
    case_type: str
    category: str
    prompt_text: str
    prompt_wav: str | None
    reference_text: str | None
    answer_text: str
    syntax_auto_ok: bool
    content_auto_ok: bool
    content_score: float
    judge_score: float | None
    judge_ok: bool
    tts_auto_ok: bool
    tts_wav_path: str | None
    tts_duration_sec: float | None
    latency_sec: float | None
    raw_answer_text: str | None
    error: str | None


@dataclass
class ServerHandle:
    process: subprocess.Popen
    stdout_path: str
    stderr_path: str
    command: list[str]


def _remove_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    filtered = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", filtered)


def normalize_text(text: str) -> str:
    text = _remove_accents(text.lower().strip())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return [tok for tok in normalize_text(text).split() if len(tok) >= 2]


def token_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = [tok for tok in tokenize(reference) if tok not in GREEK_STOPWORDS]
    hyp_tokens = [tok for tok in tokenize(hypothesis) if tok not in GREEK_STOPWORDS]
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    hyp_counts: dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    for tok in hyp_tokens:
        hyp_counts[tok] = hyp_counts.get(tok, 0) + 1
    overlap = 0
    for tok, count in ref_counts.items():
        overlap += min(count, hyp_counts.get(tok, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def greek_char_ratio(text: str) -> float:
    chars = [ch for ch in text if ch.isalpha()]
    if not chars:
        return 0.0
    greek = [
        ch
        for ch in chars
        if "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff"
    ]
    return len(greek) / len(chars)


def syntax_auto_ok(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("[ERROR:"):
        return False
    if len(tokenize(stripped)) < 3:
        return False
    return greek_char_ratio(stripped) >= 0.55


def load_text_cases_from_qa_pairs(
    qa_pairs_path: Path,
    n_per_category: int = QA_PAIRS_PER_CATEGORY,
    seed: int = EVAL_RANDOM_SEED,
) -> list[dict]:
    if not qa_pairs_path.exists():
        raise FileNotFoundError(f"QA pairs not found: {qa_pairs_path}")
    by_category: dict[str, list[dict]] = {}
    with qa_pairs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cat = item.get("category", "unknown")
            by_category.setdefault(cat, []).append(item)
    rng = random.Random(seed)
    cases: list[dict] = []
    for cat in sorted(by_category):
        pool = by_category[cat]
        rng.shuffle(pool)
        for item in pool[:n_per_category]:
            cases.append({
                "id": f"text_{cat}_{len(cases) + 1}",
                "category": cat,
                "question": item["question"].strip(),
                "reference_answer": item["answer"].strip(),
            })
    return cases


def load_audio_cases_from_voice_sentences(
    voice_sentences_path: Path,
    wavs_dir: Path,
    n_per_category: int = VOICE_CASES_PER_CATEGORY,
    seed: int = EVAL_RANDOM_SEED,
) -> list[dict]:
    if not voice_sentences_path.exists():
        raise FileNotFoundError(f"Voice sentences not found: {voice_sentences_path}")
    by_category: dict[str, list[dict]] = {}
    with voice_sentences_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            wav_path = wavs_dir / f"{item['id']}.wav"
            if not wav_path.exists():
                continue
            cat = item.get("category", "unknown")
            by_category.setdefault(cat, []).append({**item, "wav_path": str(wav_path)})
    rng = random.Random(seed)
    cases: list[dict] = []
    for cat in sorted(by_category):
        pool = by_category[cat][:]
        rng.shuffle(pool)
        for item in pool[:n_per_category]:
            cases.append({
                "id": f"audio_{cat}_{len(cases) + 1}",
                "category": cat,
                "question": item["text"].strip(),
                "reference_answer": item["text"].strip(),
                "wav_path": item["wav_path"],
            })
    return cases


def load_audio_cases_from_stt_metadata(
    metadata_path: Path,
    wavs_dir: Path,
    n_limit: int = VOICE_CASES_PER_CATEGORY * 5,
) -> list[dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"STT metadata not found: {metadata_path}")
    pool: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "|" not in line:
                continue
            stem, transcript = line.split("|", 1)
            wav_path = wavs_dir / f"{stem}.wav"
            transcript = transcript.strip()
            if not transcript or not wav_path.exists():
                continue
            pool.append({"stem": stem, "transcript": transcript, "wav_path": str(wav_path)})
    random.Random(EVAL_RANDOM_SEED).shuffle(pool)
    cases = [
        {
            "id": f"audio_stt_{i + 1}",
            "category": "stt_transcription",
            "question": item["transcript"],
            "reference_answer": item["transcript"],
            "wav_path": item["wav_path"],
        }
        for i, item in enumerate(pool[:n_limit])
    ]
    return cases


def load_text_cases_from_curated(path: Path = CURATED_TEXT_CASES) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append({
                "id": f"text_{item.get('category', 'unk')}_{i}",
                "category": item.get("category", "unknown"),
                "question": item["question"].strip(),
                "reference_answer": item["answer"].strip(),
            })
    return cases


def load_audio_cases_from_curated(path: Path = CURATED_AUDIO_CASES) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            wav = Path(item["wav_path"])
            if not wav.exists():
                continue
            cases.append({
                "id": f"audio_{item.get('category', 'unk')}_{i}",
                "category": item.get("category", "unknown"),
                "question": item["text"].strip(),
                "reference_answer": item["text"].strip(),
                "wav_path": str(wav),
            })
    return cases


def judge_answer(question: str, reference: str, answer: str) -> float | None:
    if SKIP_JUDGE:
        return None
    """Return 1.0, 0.5, 0.0, or None (verdict could not be determined).

    Scoring rubric:
      1.0 — correct, relevant, fluent Greek, no foreign words
      0.5 — correct and relevant but grammar is awkward, minor factual gap,
            OR contains isolated foreign words (e.g. one English term)
      0.0 — factually wrong, hallucinates, off-topic, severely broken Greek,
            or answer is predominantly in a language other than Greek
      None — thinking exhausted token budget before emitting a verdict (not counted)

    Raises on timeout or connection error — caller marks the case as judge=None.
    """
    prompt = (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Model answer: {answer}\n\n"
        "Rate the model answer on four criteria:\n"
        "1. CORRECTNESS: does it address the main point without contradicting the reference?\n"
        "2. FLUENCY: is the Greek grammatically correct and natural-sounding? "
        "Score 0 if sentences are broken or read like a bad translation. Minor style variation is fine.\n"
        "3. RELEVANCE: does it actually answer the question asked?\n"
        "4. LANGUAGE PURITY: is the answer written entirely in Greek? "
        "Score 0 if the answer is mostly in English, German, or any other non-Greek language. "
        "Score 0.5 if the answer is mostly Greek but contains a few isolated foreign words or phrases. "
        "Score 1 if the answer is fully in Greek.\n\n"
        "Scoring:\n"
        "  1   — all four criteria pass\n"
        "  0.5 — correct and relevant, but Greek is awkward, has a minor factual gap, "
        "or contains isolated foreign words\n"
        "  0   — factually wrong, hallucinates, off-topic, severely broken Greek, "
        "or answer is predominantly in a non-Greek language\n\n"
        "Reply with ONLY one of: 0  0.5  1\nNo other text."
    )
    resp = requests.post(
        OLLAMA_JUDGE_URL,
        json={
            "model": OLLAMA_JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 4096},
        },
        timeout=JUDGE_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    # Strip thinking tags emitted by reasoning models (qwen3, etc.)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Match 0.5 / 0,5 (European decimal) / 1 / 0 — last match wins (model often
    # reasons before emitting the final digit)
    hits = re.findall(r"0[.,]5|[01](?:\.[05])?", text)
    if not hits:
        return None
    verdict = hits[-1].replace(",", ".")
    return float(verdict)


def find_llama_server() -> Path | None:
    env_candidate = Path(LLAMA_SERVER_EXE) if LLAMA_SERVER_EXE else None
    gemma4kids = Path(os.getenv("GEMMA4KIDS_DIR", r"C:\Users\efso office\Desktop\Gemma4Kids").strip())
    candidates = [
        env_candidate,
        gemma4kids / "release" / "win-unpacked" / "resources" / "llama-server.exe",
        gemma4kids / "piper" / "llama-server.exe",
        Path(r"C:\Program Files (x86)\Llamacpp\llama.cpp-b8929\llama-server.exe"),
        Path(r"C:\llama.cpp\llama-server.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def find_first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def guess_finetuned_paths(model_tag: str) -> list[Path]:
    guesses: list[Path] = []
    env_name = f"COMPARE_FT_{model_tag.upper()}_GGUF"
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        guesses.append(Path(env_value))

    for pattern in [
        f"output/**/*{model_tag.upper()}*.gguf",
        f"output/**/*{model_tag.lower()}*.gguf",
        f"output/**/*{model_tag.upper()}*Q4*.gguf",
        f"output/**/*{model_tag.lower()}*q4*.gguf",
    ]:
        guesses.extend(sorted(BASE.glob(pattern)))

    gguf_cache = os.getenv("GGUF_CACHE_DIR", "").strip()
    if gguf_cache:
        cache_root = Path(gguf_cache)
        guesses.extend(sorted(cache_root.glob(f"gemma4gr-{model_tag.lower()}*/**/*.gguf")))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in guesses:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def default_mmproj_path(model_key: str) -> str:
    mapping = {
        "base_e4b": Path(r"N:\GEMMA GGUF UNSLOTH\E4B\mmproj-F16.gguf"),
        "ft_e4b": Path(r"N:\.cache\huggingface\hub\gemma-4-E4B-it-GR-v2\gemma4gr-e4b-v2-mmproj.gguf"),
    }
    path = mapping.get(model_key)
    return str(path) if path is not None else ""


def default_model_specs() -> list[ModelSpec]:
    base_e4b = Path(r"N:\GEMMA GGUF UNSLOTH\E4B\gemma-4-E4B-it-UD-Q4_K_XL.gguf")
    ft_e4b = find_first_existing(
        [Path(r"N:\.cache\huggingface\hub\gemma-4-E4B-it-GR-v2\gemma4gr-e4b-v2-q4_k_m.gguf")] + guess_finetuned_paths("e4b")
    )
    return [
        ModelSpec("base_e4b", "Base E4B Q4_K_XL", str(base_e4b), default_mmproj_path("base_e4b"), "gemma"),
        ModelSpec(
            "ft_e4b",
            "Fine-tuned E4B v2 (STT+QA)",
            ft_e4b,
            default_mmproj_path("ft_e4b"),
            str(Path(r"N:\Gemma4GR\gemma-4-E4B-it-GR\chat_template.jinja")),
        ),
    ]


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server() -> None:
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            response = requests.get(LLAMA_HEALTH_URL, timeout=2)
            if response.status_code == 200:
                return
        except Exception:
            time.sleep(1.5)
    raise TimeoutError(f"llama-server did not become ready on port {LLAMA_PORT}")


def start_server(model_path: Path, mmproj_path: Path, chat_template: str, log_dir: Path) -> ServerHandle:
    llama_server = find_llama_server()
    if llama_server is None:
        raise FileNotFoundError("llama-server.exe not found. Set LLAMA_SERVER_EXE or GEMMA4KIDS_DIR.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not mmproj_path.exists():
        raise FileNotFoundError(f"mmproj not found: {mmproj_path}")
    chat_template = chat_template.strip()
    chat_template_path = Path(chat_template) if chat_template.endswith(".jinja") else None
    if chat_template_path is not None and not chat_template_path.exists():
        raise FileNotFoundError(f"chat template not found: {chat_template_path}")
    if is_port_open(LLAMA_PORT):
        raise RuntimeError(
            f"Port {LLAMA_PORT} is already in use. Close the existing llama-server before starting the comparison."
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(llama_server),
        "--model",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        str(DEFAULT_CTX_SIZE),
        "--n-gpu-layers",
        str(DEFAULT_N_GPU_LAYERS),
    ]
    if chat_template_path is not None:
        command.extend(["--chat-template-file", str(chat_template_path)])
    elif chat_template:
        command.extend(["--chat-template", chat_template])
    stdout_path = log_dir / "llama_server.stdout.log"
    stderr_path = log_dir / "llama_server.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    try:
        wait_for_server()
        return ServerHandle(
            process=proc,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            command=command,
        )
    except Exception:
        proc.terminate()
        raise


def stop_server(server: ServerHandle | None) -> None:
    if server is None:
        return
    proc = server.process
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def append_text_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def request_text_answer(question: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": DEFAULT_SEED,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    response = requests.post(LLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def request_audio_answer(question: str, wav_path: Path, instruction: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": instruction},
        ],
    })
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": DEFAULT_SEED,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    response = requests.post(LLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def synthesize_answer(answer: str, out_wav: Path, play_audio: bool) -> tuple[bool, float | None, str | None]:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    result = synthesize_text(
        answer,
        _resolve_voice_onnx(),
        out_wav,
        noise_scale=PIPER_NOISE_SCALE,
        length_scale=PIPER_LENGTH_SCALE,
        noise_w=PIPER_NOISE_W,
        sentence_silence=PIPER_SENTENCE_SILENCE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        error = stderr.strip() or stdout.strip() or "Piper synthesis failed"
        return False, None, error
    if not out_wav.exists():
        return False, None, f"Piper reported success but did not create output WAV: {out_wav}"
    info = sf.info(str(out_wav))
    duration = float(info.duration)
    if play_audio:
        _play_wav(out_wav)
    return True, duration, None


def summarize_results(results: list[CaseResult]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[CaseResult]] = {}
    for item in results:
        grouped.setdefault(item.model_key, []).append(item)

    for model_key, items in grouped.items():
        label = items[0].model_label if items else model_key
        syntax_passes = sum(1 for item in items if item.syntax_auto_ok)
        content_passes = sum(1 for item in items if item.content_auto_ok)
        judge_passes = sum(1 for item in items if item.judge_ok)
        judged = [item for item in items if item.judge_score is not None]
        tts_passes = sum(1 for item in items if item.tts_auto_ok)
        avg_content = sum(item.content_score for item in items) / max(len(items), 1)
        avg_judge = sum(item.judge_score for item in judged) / max(len(judged), 1)
        summary[model_key] = {
            "label": label,
            "cases": len(items),
            "syntax_auto_passes": syntax_passes,
            "content_auto_passes": content_passes,
            "judge_passes": judge_passes,
            "judged_cases": len(judged),
            "tts_auto_passes": tts_passes,
            "avg_content_score": round(avg_content, 3),
            "avg_judge_score": round(avg_judge, 3),
        }
    return summary


class Compare4BitApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gemma4GR 4-bit Comparison")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

        self.model_specs = default_model_specs()
        self.model_vars = {spec.key: tk.StringVar(value=spec.path) for spec in self.model_specs}
        self.mmproj_vars = {spec.key: tk.StringVar(value=spec.mmproj_path) for spec in self.model_specs}
        self.chat_template_vars = {spec.key: tk.StringVar(value=spec.chat_template) for spec in self.model_specs}
        self.qa_pairs_var = tk.StringVar(value=str(DEFAULT_QA_PAIRS))
        self.voice_sentences_var = tk.StringVar(value=str(DEFAULT_VOICE_SENTENCES))
        self.stt_metadata_var = tk.StringVar(value=str(DEFAULT_STT_METADATA))
        self.audio_mode_var = tk.StringVar(value=DEFAULT_AUDIO_MODE)
        self.play_audio_var = tk.BooleanVar(value=True)
        self.stt_only_var = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value=str(EVAL_RANDOM_SEED))
        self.cases_per_cat_var = tk.StringVar(value=str(QA_PAIRS_PER_CATEGORY))
        self.force_builtin_gemma_template_var = tk.BooleanVar(value=False)
        self.use_shared_test_template_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")
        self.current_model_var = tk.StringVar(value="-")
        self.current_case_var = tk.StringVar(value="-")
        self.progress_var = tk.StringVar(value="0 / 100")

        self.prompt_box: tk.Text
        self.reference_box: tk.Text
        self.answer_box: tk.Text
        self.log_box: tk.Text

        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.last_audio_path: Path | None = None
        self.last_results_dir: Path | None = None
        self.run_log_path: Path | None = None
        self._build()
        self.root.after(150, self._poll_events)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=2)
        outer.rowconfigure(4, weight=1)

        models_frame = ttk.LabelFrame(outer, text="Models", padding=10)
        models_frame.grid(row=0, column=0, sticky="ew")
        models_frame.columnconfigure(1, weight=1)
        models_frame.columnconfigure(4, weight=1)
        models_frame.columnconfigure(7, weight=1)

        for row, spec in enumerate(self.model_specs):
            ttk.Label(models_frame, text=spec.label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(models_frame, textvariable=self.model_vars[spec.key]).grid(
                row=row, column=1, sticky="ew", padx=6, pady=4
            )
            ttk.Button(
                models_frame,
                text="Browse",
                command=lambda key=spec.key: self._browse_model(key),
            ).grid(row=row, column=2, padx=6, pady=4)
            ttk.Label(models_frame, text="mmproj").grid(row=row, column=3, sticky="w", padx=6, pady=4)
            ttk.Entry(models_frame, textvariable=self.mmproj_vars[spec.key]).grid(
                row=row, column=4, sticky="ew", padx=6, pady=4
            )
            ttk.Button(
                models_frame,
                text="Browse",
                command=lambda key=spec.key: self._browse_mmproj(key),
            ).grid(row=row, column=5, padx=6, pady=4)
            ttk.Label(models_frame, text="chat").grid(row=row, column=6, sticky="w", padx=6, pady=4)
            ttk.Entry(models_frame, textvariable=self.chat_template_vars[spec.key]).grid(
                row=row, column=7, sticky="ew", padx=6, pady=4
            )
            ttk.Button(
                models_frame,
                text="Browse",
                command=lambda key=spec.key: self._browse_chat_template(key),
            ).grid(row=row, column=8, padx=6, pady=4)

        config_frame = ttk.LabelFrame(outer, text="Inputs", padding=10)
        config_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="QA pairs").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(config_frame, textvariable=self.qa_pairs_var).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(config_frame, text="Browse", command=self._browse_qa_pairs).grid(row=0, column=2, padx=6, pady=4)
        ttk.Label(config_frame, text="Voice sentences").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(config_frame, textvariable=self.voice_sentences_var).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(config_frame, text="Browse", command=self._browse_voice_sentences).grid(row=1, column=2, padx=6, pady=4)
        ttk.Label(config_frame, text="STT metadata").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(config_frame, textvariable=self.stt_metadata_var).grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(config_frame, text="Browse", command=self._browse_stt_metadata).grid(row=2, column=2, padx=6, pady=4)
        ttk.Label(config_frame, text="Audio mode").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(
            config_frame,
            textvariable=self.audio_mode_var,
            state="readonly",
            values=("spoken_qa", "stt_transcription"),
        ).grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(config_frame, text="Play Piper audio", variable=self.play_audio_var).grid(
            row=4, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Checkbutton(
            config_frame,
            text="STT only (skip Q&A text cases)",
            variable=self.stt_only_var,
        ).grid(row=4, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(
            config_frame,
            text="Force built-in gemma template for all models",
            variable=self.force_builtin_gemma_template_var,
        ).grid(row=5, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(
            config_frame,
            text="Use shared Gemma4GR test template for all models",
            variable=self.use_shared_test_template_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        ttk.Label(config_frame, text="Sample seed").grid(row=3, column=2, sticky="w", padx=6, pady=4)
        ttk.Entry(config_frame, textvariable=self.seed_var, width=8).grid(row=3, column=3, sticky="w", padx=6, pady=4)
        ttk.Label(config_frame, text="Cases/cat").grid(row=3, column=4, sticky="w", padx=6, pady=4)
        ttk.Spinbox(config_frame, textvariable=self.cases_per_cat_var, from_=1, to=10, width=4).grid(row=3, column=5, sticky="w", padx=6, pady=4)
        ttk.Label(
            config_frame,
            text="Run size: 2 models × (4/cat text + 4/cat audio)  |  judge: Ollama qwen3.5",
        ).grid(row=4, column=2, sticky="w", padx=6, pady=4)

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(controls, text="Start Full Run", command=self.start_run).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Stop", command=self.stop_run).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Replay Last Audio", command=self.replay_last_audio).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Open Results Folder", command=self.open_results_folder).pack(side=tk.LEFT, padx=6)
        ttk.Separator(controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)
        ttk.Label(controls, text="Progress:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(controls, textvariable=self.progress_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(controls, text="Model:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(controls, textvariable=self.current_model_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(controls, text="Case:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(controls, textvariable=self.current_case_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(controls, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT, padx=(0, 6))

        prompt_frame = ttk.LabelFrame(outer, text="Prompt / Reference / Answer", padding=10)
        prompt_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.columnconfigure(1, weight=1)
        prompt_frame.columnconfigure(2, weight=1)
        prompt_frame.rowconfigure(1, weight=1)

        ttk.Label(prompt_frame, text="Prompt").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(prompt_frame, text="Reference").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(prompt_frame, text="Answer").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self.prompt_box = tk.Text(prompt_frame, wrap=tk.WORD, font=("Segoe UI", 11), height=8)
        self.reference_box = tk.Text(prompt_frame, wrap=tk.WORD, font=("Segoe UI", 11), height=8)
        self.answer_box = tk.Text(prompt_frame, wrap=tk.WORD, font=("Segoe UI", 11), height=8)
        self.prompt_box.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.reference_box.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)
        self.answer_box.grid(row=1, column=2, sticky="nsew", padx=6, pady=4)

        log_frame = ttk.LabelFrame(outer, text="Log", padding=10)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_box = tk.Text(log_frame, wrap=tk.WORD, font=("Consolas", 10), height=8, state=tk.DISABLED)
        self.log_box.grid(row=0, column=0, sticky="nsew")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _browse_model(self, key: str) -> None:
        path = filedialog.askopenfilename(
            title="Select GGUF model",
            initialdir=str(BASE),
            filetypes=[("GGUF", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.model_vars[key].set(path)

    def _browse_qa_pairs(self) -> None:
        path = filedialog.askopenfilename(
            title="Select QA pairs JSONL",
            initialdir=str(BASE / "data"),
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self.qa_pairs_var.set(path)

    def _browse_voice_sentences(self) -> None:
        path = filedialog.askopenfilename(
            title="Select voice sentences JSONL",
            initialdir=str(BASE / "data"),
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self.voice_sentences_var.set(path)

    def _browse_stt_metadata(self) -> None:
        path = filedialog.askopenfilename(
            title="Select STT metadata",
            initialdir=str(BASE / "data"),
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.stt_metadata_var.set(path)

    def _collect_specs(self) -> list[ModelSpec]:
        specs: list[ModelSpec] = []
        for spec in self.model_specs:
            specs.append(
                ModelSpec(
                    spec.key,
                    spec.label,
                    self.model_vars[spec.key].get().strip(),
                    self.mmproj_vars[spec.key].get().strip(),
                    self.chat_template_vars[spec.key].get().strip(),
                )
            )
        return specs

    def _browse_mmproj(self, key: str) -> None:
        path = filedialog.askopenfilename(
            title="Select mmproj GGUF",
            initialdir=str(BASE),
            filetypes=[("GGUF", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.mmproj_vars[key].set(path)

    def _browse_chat_template(self, key: str) -> None:
        path = filedialog.askopenfilename(
            title="Select chat template",
            initialdir=str(BASE),
            filetypes=[("Jinja", "*.jinja"), ("All files", "*.*")],
        )
        if path:
            self.chat_template_vars[key].set(path)

    def start_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Comparison", "A run is already active.")
            return

        specs = self._collect_specs()
        if self.force_builtin_gemma_template_var.get() and self.use_shared_test_template_var.get():
            messagebox.showerror(
                "Comparison",
                "Choose only one global template override:\n"
                "- Force built-in gemma template\n"
                "- Use shared Gemma4GR test template",
            )
            return
        if self.force_builtin_gemma_template_var.get():
            specs = [
                ModelSpec(spec.key, spec.label, spec.path, spec.mmproj_path, "gemma")
                for spec in specs
            ]
        elif self.use_shared_test_template_var.get():
            specs = [
                ModelSpec(spec.key, spec.label, spec.path, spec.mmproj_path, str(SHARED_CHAT_TEMPLATE))
                for spec in specs
            ]
        missing = [spec.label for spec in specs if not spec.path or not Path(spec.path).exists()]
        if missing:
            messagebox.showerror("Comparison", "Missing model path(s):\n" + "\n".join(missing))
            return
        missing_mmproj = [spec.label for spec in specs if not spec.mmproj_path or not Path(spec.mmproj_path).exists()]
        if missing_mmproj:
            messagebox.showerror("Comparison", "Missing mmproj path(s):\n" + "\n".join(missing_mmproj))
            return
        if self.use_shared_test_template_var.get() and not SHARED_CHAT_TEMPLATE.exists():
            messagebox.showerror("Comparison", f"Shared chat template not found:\n{SHARED_CHAT_TEMPLATE}")
            return
        missing_chat = [
            spec.label
            for spec in specs
            if spec.chat_template.strip().endswith(".jinja") and not Path(spec.chat_template.strip()).exists()
        ]
        if missing_chat:
            messagebox.showerror("Comparison", "Missing chat template file(s):\n" + "\n".join(missing_chat))
            return

        qa_pairs_path = Path(self.qa_pairs_var.get().strip())
        if not qa_pairs_path.exists():
            messagebox.showerror("Comparison", f"QA pairs file not found:\n{qa_pairs_path}")
            return
        voice_sentences_path = Path(self.voice_sentences_var.get().strip())
        if self.audio_mode_var.get() == "spoken_qa" and not voice_sentences_path.exists():
            messagebox.showerror("Comparison", f"Voice sentences file not found:\n{voice_sentences_path}")
            return
        stt_metadata_path = Path(self.stt_metadata_var.get().strip())
        if self.audio_mode_var.get() == "stt_transcription" and not stt_metadata_path.exists():
            messagebox.showerror("Comparison", f"STT metadata file not found:\n{stt_metadata_path}")
            return
        if self.audio_mode_var.get() == "stt_transcription" and not GREEK_STT_PROMPT_PATH.exists():
            messagebox.showerror("Comparison", f"Greek STT prompt not found:\n{GREEK_STT_PROMPT_PATH}")
            return

        piper_exe = _resolve_piper_exe()
        voice_onnx = _resolve_voice_onnx()
        if not piper_exe.exists():
            messagebox.showerror("Comparison", f"Piper executable not found:\n{piper_exe}")
            return
        if not voice_onnx.exists():
            messagebox.showerror("Comparison", f"Piper JOY model not found:\n{voice_onnx}")
            return

        try:
            seed = int(self.seed_var.get().strip())
        except ValueError:
            messagebox.showerror("Comparison", "Sample seed must be an integer.")
            return
        try:
            cases_per_cat = max(1, int(self.cases_per_cat_var.get().strip()))
        except ValueError:
            cases_per_cat = QA_PAIRS_PER_CATEGORY

        self.stop_requested.clear()
        self.last_audio_path = None
        self.last_results_dir = None
        self.run_log_path = None
        self._append_log("")
        self._append_log(f"[RUN] Starting comparison at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(f"[RUN] Sample seed: {seed}")
        self.status_var.set("Running...")
        self.worker_thread = threading.Thread(
            target=self._run_worker,
            args=(
                specs,
                qa_pairs_path,
                voice_sentences_path,
                stt_metadata_path,
                self.audio_mode_var.get(),
                self.play_audio_var.get(),
                self.stt_only_var.get(),
                seed,
                cases_per_cat,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_run(self) -> None:
        self.stop_requested.set()
        self.status_var.set("Stopping after current step...")
        self._append_log("[RUN] Stop requested.")

    def replay_last_audio(self) -> None:
        if not self.last_audio_path or not self.last_audio_path.exists():
            messagebox.showinfo("Comparison", "No synthesized audio is available yet.")
            return
        try:
            _play_wav(self.last_audio_path)
        except Exception as exc:
            messagebox.showerror("Comparison", f"Replay failed:\n{exc}")

    def open_results_folder(self) -> None:
        if not self.last_results_dir or not self.last_results_dir.exists():
            target = RESULTS_ROOT
        else:
            target = self.last_results_dir
        os.startfile(str(target))

    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self.event_queue.get_nowait()
                if event_type == "log":
                    self._append_log(str(payload))
                    if self.run_log_path is not None:
                        append_text_log(self.run_log_path, str(payload))
                elif event_type == "status":
                    self.status_var.set(str(payload))
                elif event_type == "case":
                    self._handle_case_event(payload)
                elif event_type == "audio_path":
                    self.last_audio_path = Path(str(payload))
                elif event_type == "finished":
                    self._handle_finished_event(payload)
                elif event_type == "error":
                    self.status_var.set("Failed.")
                    messagebox.showerror("Comparison", str(payload))
                elif event_type == "progress":
                    self.progress_var.set(str(payload))
        except queue.Empty:
            pass
        self.root.after(150, self._poll_events)

    def _handle_case_event(self, payload: object) -> None:
        data = dict(payload)  # type: ignore[arg-type]
        self.current_model_var.set(data.get("model_label", "-"))
        self.current_case_var.set(data.get("case_id", "-"))
        self._set_text(self.prompt_box, data.get("prompt_text", ""))
        self._set_text(self.reference_box, data.get("reference_text", ""))
        answer = data.get("answer_text", "")
        if answer:
            self._set_text(self.answer_box, answer)

    def _handle_finished_event(self, payload: object) -> None:
        data = dict(payload)  # type: ignore[arg-type]
        self.last_results_dir = Path(str(data["results_dir"]))
        self.run_log_path = Path(str(data["run_log_path"]))
        self.status_var.set("Finished.")
        self._append_log(f"[DONE] Results saved to {data['summary_path']}")
        for line in data["summary_lines"]:
            self._append_log(line)

    def _run_worker(
        self,
        specs: list[ModelSpec],
        qa_pairs_path: Path,
        voice_sentences_path: Path,
        stt_metadata_path: Path,
        audio_mode: str,
        play_audio: bool,
        stt_only: bool = False,
        seed: int = EVAL_RANDOM_SEED,
        cases_per_cat: int = QA_PAIRS_PER_CATEGORY,
    ) -> None:
        server_handle: ServerHandle | None = None
        results: list[CaseResult] = []
        try:
            if not stt_only:
                if CURATED_TEXT_CASES.exists():
                    text_cases = load_text_cases_from_curated(CURATED_TEXT_CASES)
                    self.event_queue.put(("log", f"[RUN] Curated text cases: {CURATED_TEXT_CASES} ({len(text_cases)} cases)"))
                else:
                    text_cases = load_text_cases_from_qa_pairs(qa_pairs_path, n_per_category=cases_per_cat, seed=seed)
                    self.event_queue.put(("log", f"[RUN] QA pairs: {qa_pairs_path} ({len(text_cases)} text cases)"))
            else:
                text_cases = []
            if audio_mode == "stt_transcription":
                audio_cases = load_audio_cases_from_stt_metadata(stt_metadata_path, VOICE_WAVS_DIR)
                self.event_queue.put(("log", f"[RUN] STT metadata: {stt_metadata_path} ({len(audio_cases)} audio cases)"))
            elif CURATED_AUDIO_CASES.exists():
                audio_cases = load_audio_cases_from_curated(CURATED_AUDIO_CASES)
                self.event_queue.put(("log", f"[RUN] Curated audio cases: {CURATED_AUDIO_CASES} ({len(audio_cases)} cases)"))
            else:
                audio_cases = load_audio_cases_from_voice_sentences(voice_sentences_path, VOICE_WAVS_DIR, n_per_category=cases_per_cat, seed=seed)
                self.event_queue.put(("log", f"[RUN] Voice sentences: {voice_sentences_path} ({len(audio_cases)} audio cases)"))
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = RESULTS_ROOT / run_stamp
            run_dir.mkdir(parents=True, exist_ok=True)
            run_log_path = run_dir / "run.log"
            self.run_log_path = run_log_path
            total_cases = len(specs) * (len(text_cases) + len(audio_cases))
            completed = 0
            self.event_queue.put(("log", f"[RUN] Audio mode: {audio_mode}"))
            self.event_queue.put(("log", f"[RUN] Results dir: {run_dir}"))
            self.event_queue.put(("log", f"[RUN] Judge model: {OLLAMA_JUDGE_MODEL}"))

            for spec in specs:
                if self.stop_requested.is_set():
                    break

                model_path = Path(spec.path)
                mmproj_path = Path(spec.mmproj_path)
                self.event_queue.put(("status", f"Starting {spec.label}"))
                self.event_queue.put(("log", f"[MODEL] Starting server for {spec.label}"))
                self.event_queue.put(("log", f"[MODEL]   gguf: {model_path}"))
                self.event_queue.put(("log", f"[MODEL] mmproj: {mmproj_path}"))
                self.event_queue.put(("log", f"[MODEL]   chat: {spec.chat_template}"))
                model_out_dir = run_dir / spec.key
                model_out_dir.mkdir(parents=True, exist_ok=True)
                server_handle = start_server(model_path, mmproj_path, spec.chat_template, model_out_dir)
                self.event_queue.put(("log", f"[MODEL] command: {' '.join(server_handle.command)}"))
                self.event_queue.put(("log", f"[MODEL] stdout: {server_handle.stdout_path}"))
                self.event_queue.put(("log", f"[MODEL] stderr: {server_handle.stderr_path}"))
                self.event_queue.put(("log", f"[MODEL] {spec.label} ready"))

                try:
                    for case in text_cases:
                        if self.stop_requested.is_set():
                            break
                        case_result = self._run_text_case(spec, case, model_out_dir, play_audio)
                        results.append(case_result)
                        completed += 1
                        self.event_queue.put(("progress", f"{completed} / {total_cases}"))

                    for case in audio_cases:
                        if self.stop_requested.is_set():
                            break
                        case_result = self._run_audio_case(spec, case, model_out_dir, play_audio, audio_mode)
                        results.append(case_result)
                        completed += 1
                        self.event_queue.put(("progress", f"{completed} / {total_cases}"))
                finally:
                    stop_server(server_handle)
                    server_handle = None
                    self.event_queue.put(("log", f"[MODEL] {spec.label} stopped"))

            summary = summarize_results(results)
            summary_path = run_dir / "summary.json"
            payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "llama_port": LLAMA_PORT,
                "qa_pairs_path": str(qa_pairs_path),
                "voice_sentences_path": str(voice_sentences_path),
                "stt_metadata_path": str(stt_metadata_path),
                "audio_mode": audio_mode,
                "run_log_path": str(run_log_path),
                "judge_model": OLLAMA_JUDGE_MODEL,
                "piper_voice": str(_resolve_voice_onnx()),
                "models": [asdict(spec) for spec in specs],
                "summary": summary,
                "results": [asdict(item) for item in results],
            }
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            summary_lines = ["[SUMMARY]"]
            for spec in specs:
                row = summary.get(spec.key)
                if not row:
                    continue
                summary_lines.append(
                    f"  {row['label']}: syntax {row['syntax_auto_passes']}/{row['cases']} | "
                    f"f1 {row['content_auto_passes']}/{row['cases']} | "
                    f"judge {row['judge_passes']}/{row['judged_cases']} | "
                    f"tts {row['tts_auto_passes']}/{row['cases']} | "
                    f"avg-f1 {row['avg_content_score']} | "
                    f"avg-judge {row['avg_judge_score']}"
                )

            self.event_queue.put(
                (
                    "finished",
                    {
                        "results_dir": str(run_dir),
                        "run_log_path": str(run_log_path),
                        "summary_path": str(summary_path),
                        "summary_lines": summary_lines,
                    },
                )
            )
        except Exception as exc:
            self.event_queue.put(("log", "[ERROR] Unhandled exception in comparison worker"))
            self.event_queue.put(("log", format_exception(exc)))
            self.event_queue.put(("error", str(exc)))
        finally:
            stop_server(server_handle)

    def _run_text_case(
        self,
        spec: ModelSpec,
        case: dict[str, object],
        model_out_dir: Path,
        play_audio: bool,
    ) -> CaseResult:
        question = str(case["question"])
        case_id = str(case["id"])
        category = str(case.get("category", ""))
        reference_answer = str(case["reference_answer"])
        self.event_queue.put(("status", f"{spec.label} -> {case_id}"))
        self.event_queue.put((
            "case",
            {"model_label": spec.label, "case_id": case_id,
             "prompt_text": question, "reference_text": reference_answer, "answer_text": ""},
        ))

        try:
            started_at = time.perf_counter()
            answer = request_text_answer(question)
            latency_sec = round(time.perf_counter() - started_at, 3)
            content_score = token_f1(reference_answer, answer)
            content_ok = content_score >= 0.12
            syntax_ok = syntax_auto_ok(answer)
            judge_s = judge_answer(question, reference_answer, answer)
            tts_path = model_out_dir / f"{case_id}_tts.wav"
            tts_ok, duration, tts_error = synthesize_answer(answer, tts_path, play_audio)
            if tts_ok:
                self.event_queue.put(("audio_path", str(tts_path)))
            else:
                self.event_queue.put(("log", f"[TTS] {spec.label} {case_id} error: {tts_error}"))
            result = CaseResult(
                model_key=spec.key, model_label=spec.label, case_id=case_id,
                case_type="text", category=category,
                prompt_text=question, prompt_wav=None, reference_text=reference_answer,
                answer_text=answer, syntax_auto_ok=syntax_ok,
                content_auto_ok=content_ok, content_score=round(content_score, 3),
                judge_score=judge_s, judge_ok=(judge_s is not None and judge_s >= 0.5),
                tts_auto_ok=tts_ok,
                tts_wav_path=str(tts_path) if tts_ok else None,
                tts_duration_sec=round(duration, 3) if duration is not None else None,
                latency_sec=latency_sec, raw_answer_text=answer, error=tts_error,
            )
            self.event_queue.put((
                "case",
                {"model_label": spec.label, "case_id": case_id,
                 "prompt_text": question, "reference_text": reference_answer, "answer_text": answer},
            ))
            self.event_queue.put(("log", f"[ANSWER] {spec.label} {case_id}: {answer}"))
            self.event_queue.put((
                "log",
                f"[CASE] {spec.label} {case_id} [{category}]: syntax={result.syntax_auto_ok} "
                f"f1={result.content_score} judge={judge_s} tts={result.tts_auto_ok} latency={result.latency_sec}s",
            ))
            return result
        except Exception as exc:
            result = CaseResult(
                model_key=spec.key, model_label=spec.label, case_id=case_id,
                case_type="text", category=category,
                prompt_text=question, prompt_wav=None, reference_text=reference_answer,
                answer_text=f"[ERROR: {exc}]", syntax_auto_ok=False,
                content_auto_ok=False, content_score=0.0,
                judge_score=None, judge_ok=False,
                tts_auto_ok=False, tts_wav_path=None, tts_duration_sec=None,
                latency_sec=None, raw_answer_text=None, error=str(exc),
            )
            self.event_queue.put((
                "case",
                {"model_label": spec.label, "case_id": case_id,
                 "prompt_text": question, "reference_text": reference_answer, "answer_text": result.answer_text},
            ))
            self.event_queue.put(("log", f"[CASE] {spec.label} {case_id} failed: {exc}"))
            self.event_queue.put(("log", format_exception(exc)))
            return result

    def _run_audio_case(
        self,
        spec: ModelSpec,
        case: dict[str, str],
        model_out_dir: Path,
        play_audio: bool,
        audio_mode: str,
    ) -> CaseResult:
        question = case["question"]
        case_id = case["id"]
        category = case.get("category", "")
        reference_answer = case["reference_answer"]
        wav_path = Path(case["wav_path"])
        if audio_mode == "stt_transcription":
            instruction = GREEK_STT_PROMPT_PATH.read_text(encoding="utf-8").strip()
            active_system_prompt = STT_SYSTEM_PROMPT
        else:
            instruction = AUDIO_PROMPT_INSTRUCTION
            active_system_prompt = SYSTEM_PROMPT
        self.event_queue.put(("status", f"{spec.label} -> {case_id}"))
        self.event_queue.put((
            "case",
            {"model_label": spec.label, "case_id": case_id,
             "prompt_text": question, "reference_text": reference_answer, "answer_text": ""},
        ))

        try:
            started_at = time.perf_counter()
            answer = request_audio_answer(question, wav_path, instruction, active_system_prompt)
            latency_sec = round(time.perf_counter() - started_at, 3)
            content_score = token_f1(reference_answer, answer)
            content_ok = content_score >= 0.12
            syntax_ok = syntax_auto_ok(answer)
            judge_s = judge_answer(question, reference_answer, answer)
            tts_path = model_out_dir / f"{case_id}_tts.wav"
            tts_ok, duration, tts_error = synthesize_answer(answer, tts_path, play_audio)
            if tts_ok:
                self.event_queue.put(("audio_path", str(tts_path)))
            else:
                self.event_queue.put(("log", f"[TTS] {spec.label} {case_id} error: {tts_error}"))
            result = CaseResult(
                model_key=spec.key, model_label=spec.label, case_id=case_id,
                case_type="audio", category=category,
                prompt_text=question, prompt_wav=str(wav_path), reference_text=reference_answer,
                answer_text=answer, syntax_auto_ok=syntax_ok,
                content_auto_ok=content_ok, content_score=round(content_score, 3),
                judge_score=judge_s, judge_ok=(judge_s is not None and judge_s >= 0.5),
                tts_auto_ok=tts_ok,
                tts_wav_path=str(tts_path) if tts_ok else None,
                tts_duration_sec=round(duration, 3) if duration is not None else None,
                latency_sec=latency_sec, raw_answer_text=answer, error=tts_error,
            )
            self.event_queue.put((
                "case",
                {"model_label": spec.label, "case_id": case_id,
                 "prompt_text": question, "reference_text": reference_answer, "answer_text": answer},
            ))
            self.event_queue.put(("log", f"[ANSWER] {spec.label} {case_id}: {answer}"))
            self.event_queue.put((
                "log",
                f"[CASE] {spec.label} {case_id} [{category}]: syntax={result.syntax_auto_ok} "
                f"f1={result.content_score} judge={judge_s} tts={result.tts_auto_ok} latency={result.latency_sec}s",
            ))
            return result
        except Exception as exc:
            result = CaseResult(
                model_key=spec.key, model_label=spec.label, case_id=case_id,
                case_type="audio", category=category,
                prompt_text=question, prompt_wav=str(wav_path), reference_text=reference_answer,
                answer_text=f"[ERROR: {exc}]", syntax_auto_ok=False,
                content_auto_ok=False, content_score=0.0,
                judge_score=None, judge_ok=False,
                tts_auto_ok=False, tts_wav_path=None, tts_duration_sec=None,
                latency_sec=None, raw_answer_text=None, error=str(exc),
            )
            self.event_queue.put((
                "case",
                {"model_label": spec.label, "case_id": case_id,
                 "prompt_text": question, "reference_text": reference_answer, "answer_text": result.answer_text},
            ))
            self.event_queue.put(("log", f"[CASE] {spec.label} {case_id} failed: {exc}"))
            self.event_queue.put(("log", format_exception(exc)))
            return result


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.option_add("*Font", "{Segoe UI} 10")
    Compare4BitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
