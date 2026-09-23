"""
Phase 2 Step G: merge STT-QA and text-QA LoRA adapters, export GGUF.

Strategy:
  1. Load base + STT adapter via FastModel.from_pretrained (Unsloth handles
     Gemma4ClippableLinear natively — no PEFT injection needed).
  2. Apply QA adapter via PeftModel on top (QA targets language layers only,
     no ClippableLinear — standard PEFT merge works fine).
  3. save_pretrained() the merged model to a real HF dir, then convert THAT
     with llama.cpp convert_hf_to_gguf.py + llama-quantize.

DO NOT go back to `save_pretrained_gguf` on the merged model (2026-07-17):
PEFT's merge_and_unload() returns a plain model, so Unsloth's PEFT check fails
("Unsloth: Model is not a PEFT model. Using existing checkpoint at ...") and it
silently converts the BASE checkpoint instead, discarding the merge. It writes
that base GGUF into the base snapshot dir, so this script's own consolidation
step then copied untuned weights out under a tuned name. Every merge summary on
disk records `gguf_exported: false` from this. See
GGUF_EXPORT_FALLBACK_INVESTIGATION.md.

Export now costs disk (an fp16 HF copy + an f16 GGUF, ~15 GB each for E4B)
rather than RAM. That trade is deliberate: correctness over peak memory.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from env_bootstrap import ensure_unsloth_runtime
from training.gguf_export_utils import (
    ExportValidationError,
    adapter_records,
    file_record,
    gguf_general_name,
    make_staging_dir,
    promote_directories,
    validate_adapter_plan,
    validate_quantization_methods,
    validate_staged_ggufs,
)

if __name__ == "__main__":
    ensure_unsloth_runtime(BASE)

import torch
from dotenv import load_dotenv

load_dotenv()


def _resolve_output_root() -> Path:
    root = os.getenv("GEMMA4GR_OUTPUT_ROOT", "").strip()
    if root:
        root_path = Path(root)
        return root_path if root_path.is_absolute() else BASE / root_path
    return BASE / "output"

_MODEL             = os.environ.get("MERGE_MODEL", "e2b").lower().strip()
OUTPUT_ROOT        = _resolve_output_root()
_stt_adapter_override = os.environ.get("MERGE_STT_ADAPTER_DIR", "").strip()
STT_ADAPTER = Path(_stt_adapter_override) if _stt_adapter_override else OUTPUT_ROOT / f"{_MODEL}_stt_qa" / "lora_adapter"
if not STT_ADAPTER.is_absolute():
    STT_ADAPTER = BASE / STT_ADAPTER
_qa_adapter_override = os.environ.get("MERGE_QA_ADAPTER_DIR", "").strip()
QA_ADAPTER = Path(_qa_adapter_override) if _qa_adapter_override else OUTPUT_ROOT / f"{_MODEL}_greek_qa" / "lora_adapter"
if not QA_ADAPTER.is_absolute():
    QA_ADAPTER = BASE / QA_ADAPTER
MERGED_MODEL_DIR   = OUTPUT_ROOT / "merged_model"
_gguf_cache        = os.environ.get("GGUF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "hub"))
_output_suffix     = os.environ.get("MERGE_OUTPUT_SUFFIX", "")
GGUF_DIR           = Path(_gguf_cache) / f"gemma4gr-{_MODEL}{_output_suffix}"
MERGE_SUMMARY_PATH = OUTPUT_ROOT / f"merge_summary_{_MODEL}.json"
_final_dir_override = os.environ.get("MERGE_FINAL_DIR", "").strip()
FINAL_DIR          = (
    Path(_final_dir_override)
    if _final_dir_override
    else Path(_gguf_cache) / f"gemma-4-{_MODEL.upper()}-it-GR{_output_suffix}"
)

HF_TOKEN           = os.getenv("HF_TOKEN", "")
MAX_SEQ_LEN        = int(os.getenv("QA_TRAIN_MAX_SEQ_LEN", "4096"))
EXPORT_GGUF        = os.getenv("MERGE_ADAPTERS_EXPORT_GGUF", "1").strip() == "1"
MERGE_SKIP_STT     = os.getenv("MERGE_SKIP_STT", "0").strip().lower() in ("1", "true", "yes")
MERGE_SKIP_QA      = os.getenv("MERGE_SKIP_QA", "0").strip().lower() in ("1", "true", "yes")
GGUF_QUANT_METHODS = [
    s.strip()
    for s in os.getenv("MERGE_ADAPTERS_GGUF_QUANT", "q4_k_m,q8_0").split(",")
    if s.strip()
]

# Merged fp16 HF checkpoint — the thing we actually convert. Distinct from
# GGUF_DIR so a stale GGUF can never masquerade as a fresh conversion.
MERGED_HF_DIR      = Path(
    os.environ.get("MERGE_HF_OUT_DIR", "") or (Path(_gguf_cache) / f"gemma4gr-{_MODEL}{_output_suffix}-hf")
)
LLAMA_CONVERT      = Path(os.environ.get(
    "LLAMA_CONVERT_SCRIPT",
    str(BASE / "llama.cpp" / "convert_hf_to_gguf.py"),
))
LLAMA_QUANTIZE     = Path(os.environ.get(
    "LLAMA_QUANTIZE_BIN",
    str(BASE / "llama.cpp" / "build" / "bin" / "llama-quantize"),
))
BASE_SAFETENSORS   = Path(os.environ.get("MERGE_BASE_SAFETENSORS", "")) if os.environ.get("MERGE_BASE_SAFETENSORS") else None
STOCK_MMPROJ       = Path(os.environ["MERGE_STOCK_MMPROJ"]) if os.environ.get("MERGE_STOCK_MMPROJ") else None
_name_suffix = _output_suffix.strip("-_").replace("-", " ").replace("_", " ").title()
GGUF_MODEL_NAME = os.environ.get(
    "MERGE_GGUF_MODEL_NAME",
    f"Gemma4GR {_MODEL.upper()}{f' {_name_suffix}' if _name_suffix else ''}",
).strip()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def save_merge_summary(summary: dict) -> None:
    MERGE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_step(cmd: list[str], label: str) -> None:
    """Run a subprocess, streaming combined output; raise with its output tail."""
    from collections import deque
    import subprocess

    print(f"  $ {' '.join(str(c) for c in cmd)}")
    tail: deque[str] = deque(maxlen=80)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"{label} failed (exit {returncode}):\n{''.join(tail)}")
    print(f"    {label}: ok")


def validate_tensor_outputs(gguf_stage: Path, model_files: list[str], mmproj_name: str) -> None:
    if BASE_SAFETENSORS is None:
        raise ExportValidationError("MERGE_BASE_SAFETENSORS must identify base safetensors")
    if STOCK_MMPROJ is None:
        raise ExportValidationError("MERGE_STOCK_MMPROJ must identify the stock mmproj")
    from training.gguf_not_base_guard import verify_gguf, verify_mmproj
    adapters = []
    if not MERGE_SKIP_STT:
        adapters.append(STT_ADAPTER)
    if not MERGE_SKIP_QA:
        adapters.append(QA_ADAPTER)
    for filename in model_files:
        result = verify_gguf(gguf_stage / filename, BASE_SAFETENSORS, adapters)
        if result["verdict"] != "PASS":
            raise ExportValidationError(f"tensor-level not-base guard failed for {filename}")
    mmproj_result = verify_mmproj(gguf_stage / mmproj_name, STOCK_MMPROJ, adapters)
    if mmproj_result["verdict"] != "PASS":
        raise ExportValidationError(f"mmproj tensor identity check failed: {mmproj_result['mismatched'][:5]}")


def build_release_directory(
    gguf_stage: Path,
    merged_hf_stage: Path,
    release_stage: Path,
    model_files: list[str],
    mmproj_name: str,
    manifest: dict,
) -> None:
    """Build a release using only artifacts verified in this transaction."""
    expected = [*model_files, mmproj_name]
    for filename in expected:
        src = gguf_stage / filename
        if not src.is_file():
            raise ExportValidationError(f"verified release input disappeared: {src}")
        dst = release_stage / filename
        shutil.copy2(src, dst)
        size_gb = dst.stat().st_size / 1e9
        print(f"  Staged release: {filename}  ({size_gb:.1f} GB)")

    for f in merged_hf_stage.iterdir():
        if f.is_file() and f.suffix not in {".safetensors"}:
            shutil.copy2(f, release_stage / f.name)

    system_prompt = (
        "You are a helpful assistant. You can communicate in many languages, "
        "but Greek is your strongest — you speak it with natural fluency and cultural accuracy. "
        "Always reply in the language the user writes to you in."
    )
    (release_stage / "Modelfile").write_text(
        f"FROM ./{model_files[0]}\n"
        f"ADAPTER ./{mmproj_name}\n\n"
        f'SYSTEM """\n{system_prompt}\n"""\n\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )
    (release_stage / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    actual = {path.name for path in release_stage.iterdir() if path.is_file()}
    required = set(expected) | {"Modelfile", "export_manifest.json"}
    missing = required - actual
    if missing:
        raise ExportValidationError(f"release staging is incomplete: {sorted(missing)}")
    print(f"  Release staged with {len(expected)} verified GGUFs")


def check_prerequisites_final() -> dict[str, bool]:
    status = {
        "stt_adapter": STT_ADAPTER.exists(),
        "qa_adapter": QA_ADAPTER.exists(),
    }
    print("  Adapter status:")
    stt_label = "SKIPPED (MERGE_SKIP_STT=1)" if MERGE_SKIP_STT and status["stt_adapter"] else ("FOUND" if status["stt_adapter"] else "MISSING")
    qa_label = "SKIPPED (MERGE_SKIP_QA=1)" if MERGE_SKIP_QA and status["qa_adapter"] else ("FOUND" if status["qa_adapter"] else "MISSING")
    print(f"    STT : {stt_label} - {STT_ADAPTER}")
    print(f"    QA  : {qa_label} - {QA_ADAPTER}")
    validate_adapter_plan(
        stt_exists=status["stt_adapter"],
        qa_exists=status["qa_adapter"],
        skip_stt=MERGE_SKIP_STT,
        skip_qa=MERGE_SKIP_QA,
    )
    if EXPORT_GGUF:
        validate_quantization_methods(GGUF_QUANT_METHODS)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU : {p.name} ({p.total_memory / 1e9:.1f} GB VRAM)")
    else:
        print("  GPU : none - CPU mode")
    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
    return status


# ---------------------------------------------------------------------------
# merge logic
# ---------------------------------------------------------------------------

def load_with_unsloth(FastModel, model_name: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return FastModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        full_finetuning=False,
        token=HF_TOKEN or None,
    )


def apply_qa_adapter(model, qa_adapter_path: Path):
    """Apply QA adapter via PEFT — safe because QA targets language layers only.

    E4B causes accelerate to attach CPU-offload hooks (even though the model
    is fully on GPU). These hooks break PEFT's device remapping. Strip them
    before calling PEFT so it sees a plain GPU model.
    """
    from peft import PeftModel
    from accelerate.hooks import remove_hook_from_submodules as _rm_hooks

    _rm_hooks(model)
    if hasattr(model, "hf_device_map"):
        model.hf_device_map = None
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"\n  Applying QA adapter via PEFT: {qa_adapter_path}")
    peft_model = PeftModel.from_pretrained(
        model,
        str(qa_adapter_path),
        is_trainable=False,
        autocast_adapter_dtype=False,
        device_map=None,
    )
    merged = peft_model.merge_and_unload()
    print("  QA adapter merged.")
    return merged


def export_gguf(model, processor, tokenizer) -> tuple[bool, list[str], str | None, dict | None]:
    """Save the merged model to disk, then convert THAT with llama.cpp.

    Never hand the merged model to Unsloth's save_pretrained_gguf — see the
    module docstring. Every staged GGUF must carry the explicitly requested
    positive identity before the transaction can replace published outputs.
    """
    if not EXPORT_GGUF:
        return False, [], "disabled", None

    if not LLAMA_CONVERT.exists():
        return False, [], f"converter not found: {LLAMA_CONVERT} (set LLAMA_CONVERT_SCRIPT)", None
    if not LLAMA_QUANTIZE.exists():
        return False, [], f"llama-quantize not found: {LLAMA_QUANTIZE} (set LLAMA_QUANTIZE_BIN)", None
    if not GGUF_MODEL_NAME:
        return False, [], "MERGE_GGUF_MODEL_NAME cannot be empty", None
    targets = [MERGED_HF_DIR.resolve(), GGUF_DIR.resolve(), FINAL_DIR.resolve()]
    if len(set(targets)) != len(targets):
        return False, [], "MERGED_HF_DIR, GGUF_DIR, and FINAL_DIR must be distinct", None

    stages: list[Path] = []
    try:
        merged_hf_stage = make_staging_dir(MERGED_HF_DIR)
        gguf_stage = make_staging_dir(GGUF_DIR)
        release_stage = make_staging_dir(FINAL_DIR)
        stages.extend([merged_hf_stage, gguf_stage, release_stage])

        # [1] Materialise the merged weights. This is the step whose absence
        #     caused the base-model fallback: the converter needs real files.
        print(f"\n  [1/5] Saving merged model → {merged_hf_stage}")
        # transformers' save_pretrained does:
        #     hasattr(self, "hf_device_map") and len(set(self.hf_device_map.values())) > 1
        # which explodes when the attribute is present but None. It is None here
        # because apply_qa_adapter() strips accelerate's offload hooks before the
        # PEFT merge. The model sits entirely on one device, so the multi-device
        # check is moot — drop the attribute so hasattr() is False and it skips.
        if getattr(model, "hf_device_map", "absent") is None:
            del model.hf_device_map
        model.save_pretrained(str(merged_hf_stage), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_hf_stage))
        processor.save_pretrained(str(merged_hf_stage))
        shards = sorted(merged_hf_stage.glob("*.safetensors"))
        if not shards:
            raise RuntimeError(
                f"save_pretrained wrote no safetensors into {merged_hf_stage} — "
                "nothing to convert (the merged weights never reached disk)."
            )
        total_gb = sum(s.stat().st_size for s in shards) / 1e9
        print(f"    wrote {len(shards)} shard(s), {total_gb:.1f} GB")

        stem = f"gemma4gr-{_MODEL}{_output_suffix}"
        f16_path = gguf_stage / f"{stem}.f16.gguf"

        # [2] HF -> f16 GGUF, from the merged dir only.
        print(f"\n  [2/5] Converting merged checkpoint → f16 GGUF")
        run_step(
            [sys.executable, LLAMA_CONVERT, merged_hf_stage,
             "--outfile", f16_path, "--outtype", "f16", "--model-name", GGUF_MODEL_NAME],
            "convert_hf_to_gguf",
        )
        f16_general_name = gguf_general_name(f16_path)
        if f16_general_name != GGUF_MODEL_NAME:
            raise ExportValidationError(
                f"F16 GGUF has general.name={f16_general_name!r}; expected {GGUF_MODEL_NAME!r}"
            )

        # [3] Projector conversion is required. There is no text-only fallback.
        print(f"\n  [3/5] Converting mmproj")
        mmproj_name = f"{stem}-mmproj.gguf"
        run_step(
            [sys.executable, LLAMA_CONVERT, merged_hf_stage,
             "--outfile", gguf_stage / mmproj_name, "--mmproj", "--model-name", GGUF_MODEL_NAME],
            "convert mmproj",
        )

        # [4] Quantise.
        print(f"\n  [4/5] Quantising: {', '.join(GGUF_QUANT_METHODS)}")
        model_files: list[str] = []
        for method in GGUF_QUANT_METHODS:
            filename = f"{stem}-{method}.gguf"
            model_files.append(filename)
            out = gguf_stage / filename
            run_step([LLAMA_QUANTIZE, f16_path, out, method.upper()], f"quantize {method}")

        f16_path.unlink(missing_ok=True)  # intermediate; the quants are the product
        print(f"\n  [5/5] Validating and staging release")
        gguf_records = validate_staged_ggufs(
            gguf_stage, model_files, mmproj_name, GGUF_MODEL_NAME
        )
        validate_tensor_outputs(gguf_stage, model_files, mmproj_name)
        active_adapters: dict[str, list[dict[str, object]]] = {}
        if not MERGE_SKIP_STT:
            active_adapters["stt"] = adapter_records(STT_ADAPTER)
        if not MERGE_SKIP_QA:
            active_adapters["qa"] = adapter_records(QA_ADAPTER)

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": _MODEL,
            "general_name": GGUF_MODEL_NAME,
            "base_model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
            "active_adapters": active_adapters,
            "merged_hf_shards": [
                file_record(path, relative_to=merged_hf_stage) for path in shards
            ],
            "tools": {
                "convert_hf_to_gguf": file_record(LLAMA_CONVERT),
                "llama_quantize": file_record(LLAMA_QUANTIZE),
            },
            "requested_quantizations": GGUF_QUANT_METHODS,
            "gguf_artifacts": gguf_records,
        }
        (gguf_stage / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        build_release_directory(
            gguf_stage, merged_hf_stage, release_stage,
            model_files, mmproj_name, manifest,
        )
        promote_directories([
            (merged_hf_stage, MERGED_HF_DIR),
            (gguf_stage, GGUF_DIR),
            (release_stage, FINAL_DIR),
        ])
        stages.clear()
        print(f"  Transaction committed → {FINAL_DIR}")
        return True, [*model_files, mmproj_name], None, manifest

    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"  [ERROR] GGUF export failed: {err}")
        traceback.print_exc()
        return False, [], err, None
    finally:
        for stage in stages:
            if stage.exists():
                shutil.rmtree(stage)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def merge() -> int:
    summary: dict = {
        "stt_adapter_found": STT_ADAPTER.exists(),
        "qa_adapter_found":  QA_ADAPTER.exists(),
        "merge_skip_stt":    MERGE_SKIP_STT,
        "merge_skip_qa":     MERGE_SKIP_QA,
        "gguf_dir":          str(GGUF_DIR),
        "final_dir":         str(FINAL_DIR),
        "gguf_exported":     False,
        "gguf_files":        [],
        "gguf_error":        None,
        "merge_completed":   False,
        "publication_completed": False,
        "export_manifest":   None,
    }

    try:
        status = check_prerequisites_final()
        from unsloth import FastModel, FastVisionModel

        # Step 1: load base + STT adapter (Unsloth native — handles ClippableLinear)
        # STT adapter was trained with FastVisionModel — must use it here so that
        # vision-specific params (layer_scalar etc.) are included in the device map.
        if status["stt_adapter"] and not MERGE_SKIP_STT:
            print(f"\n[1/3] Loading base model + STT adapter via Unsloth ...")
            model, processor = load_with_unsloth(FastVisionModel, str(STT_ADAPTER))
            tokenizer = get_tokenizer(processor)
            print("  STT adapter loaded and active.")
        else:
            if MERGE_SKIP_STT and status["stt_adapter"]:
                print(f"\n[1/3] MERGE_SKIP_STT=1 — skipping STT adapter, loading base model only ...")
            else:
                print(f"\n[1/3] No STT adapter — loading base model only ...")
            config_source = QA_ADAPTER if status["qa_adapter"] else STT_ADAPTER
            cfg = json.loads((config_source / "adapter_config.json").read_text(encoding="utf-8"))
            model, processor = load_with_unsloth(FastModel, cfg["base_model_name_or_path"])
            tokenizer = get_tokenizer(processor)

        # Step 2: merge QA adapter on top via PEFT (language layers only — safe)
        if status["qa_adapter"] and not MERGE_SKIP_QA:
            print(f"\n[2/3] Merging QA adapter ...")
            model = apply_qa_adapter(model, QA_ADAPTER)
        else:
            print(f"\n[2/3] MERGE_SKIP_QA=1 — QA adapter intentionally disabled.")

        # Step 3: transactional merged-HF save, GGUF conversion, and publication.
        print(f"\n[3/3] Exporting GGUF ...")
        gguf_ok, gguf_files, gguf_err, manifest = export_gguf(model, processor, tokenizer)

        summary["merge_completed"] = True
        summary["gguf_exported"]   = gguf_ok
        summary["gguf_files"]      = gguf_files
        summary["gguf_error"]      = gguf_err
        summary["publication_completed"] = gguf_ok
        summary["export_manifest"] = (
            str(GGUF_DIR / "export_manifest.json") if manifest is not None else None
        )

        print(f"\n{'=' * 60}")
        if gguf_ok:
            print("  Merge complete")
            print(f"  GGUF output : {GGUF_DIR}")
            for name in gguf_files:
                size = (GGUF_DIR / name).stat().st_size / 1e9
                print(f"    {name}  ({size:.1f} GB)")
            print(f"  Final dir   : {FINAL_DIR}")
            print(f"{'=' * 60}")
            return 0

        if gguf_err == "disabled":
            print("  Merge complete — GGUF export explicitly disabled")
            print("  No publication was attempted.")
            print(f"{'=' * 60}")
            return 0

        # A failed export used to exit 0 and print "GGUF: skipped", while
        # consolidate_output() had already copied base weights into FINAL_DIR.
        # Fail loudly instead: nothing was consolidated, and callers can tell.
        print("  MERGE FAILED — no GGUF exported")
        print(f"  Reason      : {gguf_err}")
        print("  Nothing was written to the final output dir.")
        print(f"{'=' * 60}")
        return 1

    except Exception as exc:
        summary["merge_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        summary["gguf_error"] = summary["gguf_error"] or summary["merge_error"]
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()
        return 1
    finally:
        save_merge_summary(summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Gemma4GR Phase 2 — Merge Adapters ({_MODEL.upper()})")
    print("=" * 60 + "\n")
    raise SystemExit(merge())
