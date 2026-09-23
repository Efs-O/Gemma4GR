"""Deterministic, resumable Gemma4GR baseline evaluation helpers and runner."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import string
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "v3" / "eval"
OUT = ROOT / "private" / "eval"
SYSTEM = "Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. Μην χρησιμοποιείς Αγγλικά, Ρωσικά ή άλλη γλώσσα, εκτός αν το ζητά ρητά η ερώτηση. Δώσε άμεση, ουσιαστική και πλήρη απάντηση."


def _garbage_pattern():
    """Import the canonical pattern rather than duplicating its definition."""
    import importlib.util
    p = ROOT / "scripts" / "build_v3_dataset.py"
    spec = importlib.util.spec_from_file_location("build_v3_dataset", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    for name in ("GARB", "GARBAGE_IN_WORD", "GARBAGE_RE", "GARBAGE_PATTERN"):
        value = getattr(mod, name, None)
        if hasattr(value, "search"):
            return value
    raise RuntimeError("canonical garbage-in-word regex not found")


def _f1(c_ref: Counter, c_hyp: Counter) -> float:
    overlap = sum((c_ref & c_hyp).values())
    if not c_ref and not c_hyp:
        return 1.0
    if not overlap:
        return 0.0
    p, r = overlap / sum(c_hyp.values()), overlap / sum(c_ref.values())
    return 2 * p * r / (p + r)


def chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    scores = []
    for size in range(1, n + 1):
        a = Counter(reference[i:i + size] for i in range(max(0, len(reference) - size + 1)))
        b = Counter(hypothesis[i:i + size] for i in range(max(0, len(hypothesis) - size + 1)))
        overlap = sum((a & b).values())
        if not a and not b:
            scores.append(1.0)
        elif not overlap:
            scores.append(0.0)
        else:
            p, r = overlap / sum(b.values()), overlap / sum(a.values())
            scores.append((1 + beta * beta) * p * r / (beta * beta * p + r))
    return sum(scores) / len(scores)


def token_f1(reference: str, hypothesis: str) -> float:
    norm = lambda s: re.findall(r"\w+", s.casefold(), flags=re.UNICODE)
    return _f1(Counter(norm(reference)), Counter(norm(hypothesis)))


def greek_share(text: str) -> float:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    return sum("GREEK" in unicodedata.name(c, "") for c in letters) / len(letters) if letters else 0.0


def garbage_fragments(text: str) -> list[str]:
    pattern = _garbage_pattern()
    matches = [m.group(0) for m in pattern.finditer(text)]
    exotic = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u052f\u0600-\u06ff\u0590-\u05ff\u0e00-\u0e7f]")
    matches.extend(m.group(0) for m in exotic.finditer(text))
    return matches


def metrics(answer: str, reference: str | None = None) -> dict:
    garbage = garbage_fragments(answer)
    result = {"greek_script_share": greek_share(answer), "garbage": bool(garbage), "garbage_fragments": garbage, "length_chars": len(answer)}
    if reference is not None:
        result.update(chrf=chrf(reference, answer), token_f1=token_f1(reference, answer))
    return result


def parse_rendered(text: str) -> tuple[list[dict], str]:
    text = re.sub(r"^<bos>", "", text)
    turns = re.findall(r"<\|turn>(system|user|model)\n(.*?)<turn\|>", text, re.S)
    msgs = [{"role": {"model": "assistant"}.get(role, role), "content": content} for role, content in turns]
    ref = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    msgs = [m for m in msgs if m["role"] != "assistant"]
    if not msgs or msgs[0]["role"] != "system" or msgs[0]["content"] != SYSTEM:
        raise ValueError("eval row system prompt differs from training system prompt")
    return msgs, ref


def verify_audio_sha(path: Path, expected_sha: str) -> bytes:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"audio sha256 mismatch: {path}")
    return raw


def audio_bytes(path: Path, expected_sha: str) -> str:
    raw = verify_audio_sha(path, expected_sha)
    import soundfile as sf
    import scipy.signal
    samples, sr = sf.read(__import__("io").BytesIO(raw), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        mono = scipy.signal.resample_poly(mono, 16000 // divisor, sr // divisor)
    buf = __import__("io").BytesIO()
    sf.write(buf, mono, 16000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def request_payload(messages: list[dict], set_name: str) -> dict:
    return {"model": "local", "messages": messages, "temperature": 0, "top_p": 1, "seed": 42,
            "max_tokens": 256 if set_name == "audio" else 512,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}


def run_settings(model: Path, mmproj: Path, set_name: str, port: int = 8080) -> dict:
    server_command = f'"{ROOT / "private/tools/llama.cpp-b11095/llama-server.exe"}" -m "{model}" --mmproj "{mmproj}" -c 4096 -ngl 99 --seed 42 --jinja -np 1 --host 127.0.0.1 --port {port}'
    return {
        "server_command": server_command,
        "cache_prompt": False,
        "slots": 1,
        "max_tokens": 256 if set_name == "audio" else 512,
        "temperature": 0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def settings_fingerprint(settings: dict) -> str:
    canonical = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def completed_cases(path: Path, fingerprint: str) -> set[tuple[str, str]]:
    if not path.exists(): return set()
    rows = _load(path)
    if any(r.get("settings_fingerprint") != fingerprint for r in rows):
        raise ValueError(f"stale rows from a different configuration in {path}; move them aside")
    return {(r["run_key"], str(r["case_id"])) for r in rows}


def run(run_key: str, model: Path, mmproj: Path, set_name: str, start: int = 0, end: int | None = None, port: int = 8080):
    cases = _load(EVAL / {"text": "text_eval.jsonl", "probe": "garbage_probe.jsonl", "audio": "audio_eval.jsonl"}[set_name])
    cases = cases[start:end]
    target = OUT / run_key / f"{set_name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    settings = run_settings(model, mmproj, set_name, port)
    fingerprint = settings_fingerprint(settings)
    done = completed_cases(target, fingerprint)
    with target.open("a", encoding="utf-8", newline="\n") as f:
        for case_index, case in enumerate(cases, start=start + 1):
            cid = str(case.get("id", case.get("case_id", case_index)))
            if (run_key, cid) in done:
                continue
            if set_name == "text":
                messages, reference = parse_rendered(case["text"])
            elif set_name == "probe":
                messages, reference = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}], None
            else:
                reference = case["reference"]
                aud = audio_bytes(ROOT / case["audio_path"], case["audio_sha256"])
                messages = [{"role": "user", "content": [{"type": "text", "text": case["prompt"]}, {"type": "input_audio", "input_audio": {"data": aud, "format": "wav"}}]}]
            payload = request_payload(messages, set_name)
            body = json.dumps(payload).encode()
            t0 = time.perf_counter()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            response = json.loads(urllib.request.urlopen(req, timeout=900).read())
            latency = time.perf_counter() - t0
            answer = response["choices"][0]["message"]["content"] or ""
            finish = response["choices"][0].get("finish_reason")
            row = {"run_key": run_key, "case_id": cid, "prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "answer": answer, "finish_reason": finish, "latency_s": latency, "metrics": metrics(answer, reference), "stop_failure": finish != "stop", "settings_fingerprint": fingerprint, "settings": settings}
            if set_name == "audio":
                norm = lambda s: re.sub(r"[^\w\s]", "", unicodedata.normalize("NFC", s).lower()).split()
                hyp, ref = norm(answer), norm(reference)
                def edit(a, b):
                    row = list(range(len(b) + 1))
                    for i, x in enumerate(a, 1):
                        new = [i]
                        for j, y in enumerate(b, 1): new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
                        row = new
                    return row[-1]
                clean_chars = lambda s: list(re.sub(r"[^\w\s]", "", unicodedata.normalize("NFC", s).lower()))
                hyp_chars, ref_chars = clean_chars(answer), clean_chars(reference)
                row["metrics"].update(wer=edit(hyp, ref) / max(1, len(ref)), cer=edit(hyp_chars, ref_chars) / max(1, len(ref_chars)))
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-key"); p.add_argument("--model", type=Path); p.add_argument("--mmproj", type=Path)
    p.add_argument("--set", choices=["text", "probe", "audio"]); p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int); p.add_argument("--port", type=int, default=8080)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize(); return
    if not all([a.run_key, a.model, a.mmproj, a.set]): p.error("run requires --run-key --model --mmproj --set")
    run(a.run_key, a.model, a.mmproj, a.set, a.start, a.end, a.port)


def summarize():
    import random
    import statistics
    def boot(values, seed=3407, count=1000):
        if not values: return [None, None, None]
        rng = random.Random(seed)
        means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(count)]
        means.sort()
        return [statistics.mean(values), means[24], means[974]]
    allruns = {}
    rawruns = {}
    for path in sorted(OUT.glob("*/*.jsonl")):
        rows = _load(path)
        if not rows: continue
        key = (path.parent.name, path.stem)
        rawruns[key] = rows
        cols = {}
        for field in ("chrf", "cer", "garbage"):
            present = [r["metrics"].get(field) for r in rows]
            vals = [float(v) if field != "garbage" else float(bool(v)) for v in present if v is not None]
            cols[field] = boot(vals) if vals else [None, None, None]
        allruns[f"{key[0]}/{key[1]}"] = {"count": len(rows), "metrics": cols, "stop_failures": sum(bool(r["stop_failure"]) for r in rows), "mean_length_chars": statistics.mean(r["metrics"]["length_chars"] for r in rows)}
    paired = {}
    for (run_key, set_name), rows in rawruns.items():
        quant = run_key.removeprefix("v2fixed_").removeprefix("stock_")
        if run_key.startswith("stock_"): continue
        baseline = rawruns.get((f"stock_{quant}", set_name))
        if baseline is None: continue
        left = {str(r["case_id"]): r for r in baseline}
        right = {str(r["case_id"]): r for r in rows}
        wins = ties = losses = 0
        for case_id in left.keys() & right.keys():
            a = left[case_id]["metrics"].get("chrf", 0)
            b = right[case_id]["metrics"].get("chrf", 0)
            wins += b > a; losses += b < a; ties += b == a
        paired[f"{run_key}/{set_name} vs stock_{quant}"] = {"wins": wins, "ties": ties, "losses": losses}
    result = {"runs": allruns, "paired_chrf_vs_stock": paired}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| Run / set | n | chrF mean [95% CI] | CER mean [95% CI] | garbage mean [95% CI] | stop failures | mean chars |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in allruns.items():
        c = v["metrics"]
        fmt = lambda x: "—" if x[0] is None else f"{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]"
        lines.append(f"| {k} | {v['count']} | {fmt(c['chrf'])} | {fmt(c['cer'])} | {fmt(c['garbage'])} | {v['stop_failures']} | {v['mean_length_chars']:.1f} |")
    if paired:
        lines += ["", "| Paired comparison | wins | ties | losses |", "|---|---:|---:|---:|"]
        lines += [f"| {k} | {v['wins']} | {v['ties']} | {v['losses']} |" for k, v in paired.items()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
