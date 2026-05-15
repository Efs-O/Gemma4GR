"""
Phase 2 Step G: merge STT-QA and text-QA LoRA adapters, export GGUF.

Strategy (minimum RAM):
  1. Load base + STT adapter via FastModel.from_pretrained (Unsloth handles
     Gemma4ClippableLinear natively — no PEFT injection needed).
  2. Apply QA adapter via PeftModel on top (QA targets language layers only,
     no ClippableLinear — standard PEFT merge works fine).
  3. Call save_pretrained_gguf directly — quantises on the fly, never
     materialises a full fp16 model in system RAM.

Peak memory (E2B): ~8 GB VRAM, ~2–3 GB system RAM.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
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
_gguf_cache        = os.environ.get("GGUF_CACHE_DIR", r"N:\.cache\huggingface\hub")
_output_suffix     = os.environ.get("MERGE_OUTPUT_SUFFIX", "")
GGUF_DIR           = Path(_gguf_cache) / f"gemma4gr-{_MODEL}{_output_suffix}"
MERGE_SUMMARY_PATH = OUTPUT_ROOT / f"merge_summary_{_MODEL}.json"
FINAL_DIR          = Path(_gguf_cache) / f"gemma-4-{_MODEL.upper()}-it-GR{_output_suffix}"

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
MAX_MEM_USAGE      = float(os.getenv("MERGE_MAX_MEM_USAGE", "0.75"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def save_merge_summary(summary: dict) -> None:
    MERGE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_modelfile(gguf_dir: Path) -> None:
    q4_file = next(
        (f for f in sorted(gguf_dir.iterdir()) if "q4" in f.name.lower() and f.suffix == ".gguf"),
        None,
    )
    if not q4_file:
        return
    (gguf_dir / "Modelfile").write_text(
        f"FROM ./{q4_file.name}\n\n"
        'SYSTEM """\n'
        "You are a helpful assistant who always replies in Greek using natural and correct language.\n"
        '"""\n\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )
    print(f"  Modelfile written: {gguf_dir / 'Modelfile'}")


def consolidate_output() -> None:
    """Gather scattered GGUFs + tokenizer into one clean FINAL_DIR."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"gemma4gr-{_MODEL}{_output_suffix}"

    # Unsloth writes Q4/Q8 GGUFs into the base model's snapshot dir, not GGUF_DIR.
    # Search both GGUF_DIR siblings and the base model snapshot tree.
    search_dirs = [GGUF_DIR, Path(str(GGUF_DIR) + "_gguf")]
    base_snapshot_root = Path(_gguf_cache) / f"models--unsloth--gemma-4-{_MODEL.upper()}-it" / "snapshots"
    if base_snapshot_root.exists():
        search_dirs.append(base_snapshot_root)

    found: dict[str, Path | None] = {"q4_k_m": None, "q8_0": None, "mmproj": None}
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.gguf"):
            n = f.name.lower()
            if "q4_k_m" in n and found["q4_k_m"] is None:
                found["q4_k_m"] = f
            elif "q8_0" in n and found["q8_0"] is None:
                found["q8_0"] = f
            elif "mmproj" in n and found["mmproj"] is None:
                found["mmproj"] = f

    name_map = {
        "q4_k_m": f"{prefix}-q4_k_m.gguf",
        "q8_0":   f"{prefix}-q8_0.gguf",
        "mmproj": f"{prefix}-mmproj.gguf",
    }
    for key, src in found.items():
        if src is None:
            print(f"  [WARN] consolidate: {key} GGUF not found")
            continue
        dst = FINAL_DIR / name_map[key]
        shutil.copy2(src, dst)
        size_gb = dst.stat().st_size / 1e9
        print(f"  Consolidated: {name_map[key]}  ({size_gb:.1f} GB)")

    # Copy tokenizer / processor files from GGUF_DIR
    for f in GGUF_DIR.iterdir():
        if f.suffix not in {".gguf", ".safetensors"}:
            shutil.copy2(f, FINAL_DIR / f.name)

    # Write Modelfile with ADAPTER line for mmproj
    system_prompt = (
        "You are a helpful assistant. You can communicate in many languages, "
        "but Greek is your strongest — you speak it with natural fluency and cultural accuracy. "
        "Always reply in the language the user writes to you in."
    )
    (FINAL_DIR / "Modelfile").write_text(
        f"FROM ./{name_map['q4_k_m']}\n"
        f"ADAPTER ./{name_map['mmproj']}\n\n"
        f'SYSTEM """\n{system_prompt}\n"""\n\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )
    print(f"  Modelfile written (with ADAPTER line)")
    print(f"\n  Final output dir: {FINAL_DIR}")


def check_prerequisites() -> dict[str, bool]:
    status = {
        "stt_adapter": STT_ADAPTER.exists(),
        "qa_adapter":  QA_ADAPTER.exists(),
    }
    print("  Adapter status:")
    stt_label = "SKIPPED (MERGE_SKIP_STT=1)" if MERGE_SKIP_STT and status["stt_adapter"] else ("FOUND" if status["stt_adapter"] else "MISSING")
    print(f"    STT : {stt_label} — {STT_ADAPTER}")
    print(f"    QA  : {'FOUND' if status['qa_adapter']  else 'MISSING'} — {QA_ADAPTER}")
    if not any(status.values()):
        print("\n[ERROR] No adapters found — run training steps first.")
        raise SystemExit(1)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU : {p.name} ({p.total_memory / 1e9:.1f} GB VRAM)")
    else:
        print("  GPU : none — CPU mode")
    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
    return status


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
    if not any(status.values()):
        print("\n[ERROR] No adapters found - run training steps first.")
        raise SystemExit(1)
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

    try:
        _rm_hooks(model)
    except Exception:
        pass
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


def export_gguf(model, processor, tokenizer) -> tuple[bool, list[str], str | None]:
    if not EXPORT_GGUF:
        return False, [], "disabled"

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    # Save processor first so the GGUF converter finds preprocessor_config.json
    # (needed for image_mean/image_std in the mmproj conversion step).
    processor.save_pretrained(str(GGUF_DIR))
    print(f"\n  Exporting GGUF → {GGUF_DIR}")
    print(f"  Quantization : {', '.join(GGUF_QUANT_METHODS)}")
    print(f"  Max mem usage: {MAX_MEM_USAGE}")

    try:
        model.save_pretrained_gguf(
            str(GGUF_DIR),
            tokenizer,
            quantization_method=GGUF_QUANT_METHODS,
            maximum_memory_usage=MAX_MEM_USAGE,
        )
    except TypeError:
        # older Unsloth without maximum_memory_usage param
        model.save_pretrained_gguf(
            str(GGUF_DIR),
            tokenizer,
            quantization_method=GGUF_QUANT_METHODS,
        )
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"  [WARN] GGUF export failed: {err}")
        return False, [], err

    gguf_files = sorted(f for f in GGUF_DIR.iterdir() if f.suffix.lower() == ".gguf")

    # Remove Unsloth's intermediate safetensors — not needed once GGUFs exist.
    for leftover in GGUF_DIR.glob("*.safetensors"):
        leftover.unlink()
        print(f"  Removed intermediate: {leftover.name}")

    print(f"\n  Consolidating into final output dir ...")
    consolidate_output()

    return bool(gguf_files), [f.name for f in gguf_files], None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def merge() -> int:
    status = check_prerequisites_final()
    summary: dict = {
        "stt_adapter_found": status["stt_adapter"],
        "qa_adapter_found":  status["qa_adapter"],
        "merge_skip_qa":     MERGE_SKIP_QA,
        "gguf_dir":          str(GGUF_DIR),
        "gguf_exported":     False,
        "gguf_files":        [],
        "gguf_error":        None,
        "merge_completed":   False,
    }

    try:
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
            print(f"\n[2/3] No QA adapter — skipping.")

        # Step 3: export directly to GGUF (on-the-fly quantisation, minimal RAM)
        print(f"\n[3/3] Exporting GGUF ...")
        gguf_ok, gguf_files, gguf_err = export_gguf(model, processor, tokenizer)

        summary["merge_completed"] = True
        summary["gguf_exported"]   = gguf_ok
        summary["gguf_files"]      = gguf_files
        summary["gguf_error"]      = gguf_err

        print(f"\n{'=' * 60}")
        print("  Merge complete")
        if gguf_ok:
            print(f"  GGUF output : {GGUF_DIR}")
            for name in gguf_files:
                size = (GGUF_DIR / name).stat().st_size / 1e9
                print(f"    {name}  ({size:.1f} GB)")
        else:
            print(f"  GGUF        : skipped — {gguf_err}")
        print(f"{'=' * 60}")
        return 0

    except Exception as exc:
        summary["merge_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
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
