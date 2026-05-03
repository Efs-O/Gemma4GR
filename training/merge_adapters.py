"""
Phase 2 Step G: merge STT and QA LoRA adapters into the base model.

This script always tries to produce merged Hugging Face artifacts first.
GGUF export is optional and fail-soft so a successful adapter merge is not
discarded if llama.cpp or quantization is unavailable on the machine.
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

BASE = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from env_bootstrap import ensure_unsloth_runtime

ensure_unsloth_runtime(BASE)

import torch
from dotenv import load_dotenv

load_dotenv()

STT_ADAPTER = BASE / "output" / "e2b_greek_stt" / "lora_adapter"
QA_ADAPTER = BASE / "output" / "e2b_greek_qa" / "lora_adapter"
MERGED_MODEL_DIR = BASE / "output" / "merged_model"
GGUF_DIR = BASE / "output" / "merged_gguf"
MERGE_SUMMARY_PATH = BASE / "output" / "merge_summary.json"

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")
MODEL_PATH_OVERRIDE = os.getenv("E2B_MODEL_PATH", "").strip()
MAX_SEQ_LEN = int(os.getenv("QA_TRAIN_MAX_SEQ_LEN", "4096"))
EXPORT_GGUF = os.getenv("MERGE_ADAPTERS_EXPORT_GGUF", "1").strip() == "1"
GGUF_QUANT_METHODS = [
    item.strip()
    for item in os.getenv("MERGE_ADAPTERS_GGUF_QUANT", "q4_k_m,q8_0").split(",")
    if item.strip()
]


def resolve_model_source() -> str:
    if MODEL_PATH_OVERRIDE:
        path = Path(MODEL_PATH_OVERRIDE)
        if path.exists():
            return str(path)

    default_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--gemma-4-E2B-it"
    refs_main = default_cache / "refs" / "main"
    snapshots = default_cache / "snapshots"
    if refs_main.exists() and snapshots.exists():
        snapshot_name = refs_main.read_text(encoding="utf-8", errors="replace").strip()
        snapshot = snapshots / snapshot_name
        if snapshot.exists():
            return str(snapshot)

    return MODEL_NAME


def load_unsloth_model(FastModel, dtype: torch.dtype):
    model_source = resolve_model_source()
    common_kwargs = dict(
        model_name=model_source,
        dtype=dtype,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        full_finetuning=False,
        token=HF_TOKEN or None,
    )
    try:
        return FastModel.from_pretrained(**common_kwargs)
    except Exception as exc:
        print(f"  [WARN] Initial model load failed: {exc}")
        print("  [INFO] Retrying model load from local Hugging Face cache only...")
        return FastModel.from_pretrained(
            **common_kwargs,
            local_files_only=True,
        )


def get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def adapter_targets(status: dict[str, bool]) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    if status["stt_adapter"]:
        targets.append(("STT", STT_ADAPTER))
    if status["qa_adapter"]:
        targets.append(("QA", QA_ADAPTER))
    return targets


def check_prerequisites() -> dict[str, bool]:
    status = {
        "stt_adapter": STT_ADAPTER.exists(),
        "qa_adapter": QA_ADAPTER.exists(),
    }

    print("  Adapter status:")
    print(f"    STT adapter: {'Found' if status['stt_adapter'] else 'Missing'} - {STT_ADAPTER}")
    print(f"    QA  adapter: {'Found' if status['qa_adapter'] else 'Missing'} - {QA_ADAPTER}")

    if not status["stt_adapter"] and not status["qa_adapter"]:
        print("\n[ERROR] No adapters found. Run the training steps first.")
        raise SystemExit(1)

    if not status["stt_adapter"]:
        print("\n[WARN] STT adapter missing - will merge QA only.")
    if not status["qa_adapter"]:
        print("\n[WARN] QA adapter missing - will merge STT only.")

    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 10:
            print("  [WARN] Float16 merge may be tight on VRAM; CPU fallback is enabled.")
    else:
        print("  [INFO] No CUDA detected - merge will run on CPU.")

    print(f"  Base source: {resolve_model_source()}")
    return status


def save_merge_summary(summary: dict[str, object]) -> None:
    MERGE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def has_meta_parameters(model) -> bool:
    return any(getattr(param, "is_meta", False) for param in model.parameters())


def merge_single_adapter(model, adapter_name: str, adapter_path: Path):
    from peft import PeftModel

    print(f"\n  Applying {adapter_name} adapter from {adapter_path} ...")
    peft_model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
        autocast_adapter_dtype=False,
        ephemeral_gpu_offload=False,
        low_cpu_mem_usage=False,
    )
    if has_meta_parameters(peft_model):
        raise RuntimeError(f"{adapter_name} adapter load left parameters on the meta device.")

    merged = peft_model.merge_and_unload()
    if has_meta_parameters(merged):
        raise RuntimeError(f"{adapter_name} merge left parameters on the meta device.")
    print(f"  {adapter_name} adapter merged")
    return merged


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_merged_hf_artifacts(model, processor, tokenizer) -> None:
    prepare_output_dir(MERGED_MODEL_DIR)
    model.save_pretrained(str(MERGED_MODEL_DIR))
    processor.save_pretrained(str(MERGED_MODEL_DIR))
    tokenizer.save_pretrained(str(MERGED_MODEL_DIR))
    print(f"\n  Merged model artifacts saved to {MERGED_MODEL_DIR}")


def write_modelfile(gguf_files: list[Path]) -> None:
    q4_file = next((f for f in gguf_files if "q4" in f.name.lower()), None)
    if not q4_file:
        return

    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(
        f"FROM ./{q4_file.name}\n\n"
        "SYSTEM \"\"\"\n"
        "You are a helpful assistant who always replies in Greek using natural and correct language.\n"
        "\"\"\"\n\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )
    print(f"  Modelfile written: {modelfile_path}")


def copy_generated_files(file_paths: list[str]) -> list[Path]:
    copied: list[Path] = []
    prepare_output_dir(GGUF_DIR)

    for file_path_str in file_paths:
        source = Path(file_path_str)
        if not source.exists():
            continue
        destination = GGUF_DIR / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied.append(destination)

    return copied


def export_gguf(model, tokenizer) -> tuple[bool, list[str], str | None]:
    if not EXPORT_GGUF:
        print("\n[4/4] Skipping GGUF export by configuration.")
        return False, [], "GGUF export disabled by MERGE_ADAPTERS_EXPORT_GGUF=0."

    if not hasattr(model, "save_pretrained_gguf"):
        print("\n[4/4] GGUF export not available on this model wrapper.")
        return False, [], "Model wrapper does not expose save_pretrained_gguf."

    prepare_output_dir(GGUF_DIR)
    print(f"\n[4/4] Exporting GGUF to {GGUF_DIR} ...")
    print(f"  Quantization targets: {', '.join(GGUF_QUANT_METHODS) if GGUF_QUANT_METHODS else 'none'}")

    try:
        if GGUF_QUANT_METHODS:
            result = model.save_pretrained_gguf(
                str(GGUF_DIR),
                tokenizer,
                quantization_method=GGUF_QUANT_METHODS,
            )
        else:
            result = model.save_pretrained_gguf(str(GGUF_DIR), tokenizer)
    except Exception as exc:
        print(f"  [WARN] GGUF export failed: {exc}")
        return False, [], "".join(traceback.format_exception_only(type(exc), exc)).strip()

    generated_files: list[str] = []
    modelfile_location = None
    if isinstance(result, dict):
        generated_files = [str(path) for path in result.get("gguf_files", [])]
        modelfile_location = result.get("modelfile_location")

    copied_files = copy_generated_files(generated_files)
    gguf_files = sorted(path for path in copied_files if path.suffix.lower() == ".gguf")

    if modelfile_location:
        source_modelfile = Path(modelfile_location)
        if source_modelfile.exists():
            shutil.copy2(source_modelfile, GGUF_DIR / source_modelfile.name)
    else:
        write_modelfile(gguf_files)

    return bool(gguf_files), [f.name for f in gguf_files], None


def release_model(model) -> None:
    try:
        if model is not None:
            model.to("cpu")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_merge(status: dict[str, bool], use_cpu: bool) -> tuple[object, object, object, list[str], str]:
    from unsloth import FastModel

    phase_label = "CPU fallback" if use_cpu else "GPU"
    dtype = torch.float32 if use_cpu or not torch.cuda.is_available() else torch.float16

    print(f"\n[1/4] Loading base model in {phase_label} mode ...")
    print(f"  Source: {resolve_model_source()}")
    model, processor = load_unsloth_model(FastModel, dtype)
    tokenizer = get_tokenizer(processor)

    if use_cpu:
        model = model.to("cpu")

    merged_adapters: list[str] = []
    targets = adapter_targets(status)
    total_steps = len(targets)
    for index, (adapter_name, adapter_path) in enumerate(targets, start=1):
        print(f"\n[{index + 1}/{total_steps + 2}] Merging {adapter_name} adapter ...")
        model = merge_single_adapter(model, adapter_name, adapter_path)
        merged_adapters.append(adapter_name.lower())

    return model, processor, tokenizer, merged_adapters, "cpu" if use_cpu else "cuda"


def merge() -> int:
    status = check_prerequisites()
    summary: dict[str, object] = {
        "base_model": MODEL_NAME,
        "model_source": resolve_model_source(),
        "merged_model_dir": str(MERGED_MODEL_DIR),
        "gguf_dir": str(GGUF_DIR),
        "requested_adapters": [name for name, _ in adapter_targets(status)],
        "stt_adapter_found": status["stt_adapter"],
        "qa_adapter_found": status["qa_adapter"],
        "gguf_export_requested": EXPORT_GGUF,
        "gguf_exported": False,
        "gguf_files": [],
        "gguf_error": None,
        "merge_device": None,
        "cpu_fallback_used": False,
        "merge_completed": False,
    }

    model = None
    processor = None
    tokenizer = None
    merged_adapters: list[str] = []
    merge_error: Exception | None = None

    try:
        try:
            model, processor, tokenizer, merged_adapters, device_used = run_merge(
                status,
                use_cpu=not torch.cuda.is_available(),
            )
        except Exception as exc:
            merge_error = exc
            if not torch.cuda.is_available():
                raise

            print(f"\n  [WARN] GPU merge failed: {exc}")
            print("  [INFO] Releasing GPU state and retrying adapter merge on CPU...")
            release_model(model)
            model, processor, tokenizer, merged_adapters, device_used = run_merge(status, use_cpu=True)
            summary["cpu_fallback_used"] = True

        summary["merge_device"] = device_used
        summary["merged_adapters"] = merged_adapters

        save_merged_hf_artifacts(model, processor, tokenizer)
        summary["merge_completed"] = True

        gguf_exported, gguf_files, gguf_error = export_gguf(model, tokenizer)
        summary["gguf_exported"] = gguf_exported
        summary["gguf_files"] = gguf_files
        summary["gguf_error"] = gguf_error

        if gguf_exported:
            print(f"\n  GGUF output: {GGUF_DIR}")
            for file_name in gguf_files:
                file_path = GGUF_DIR / file_name
                size_gb = file_path.stat().st_size / 1e9
                print(f"    {file_name} ({size_gb:.1f} GB)")
        else:
            print("\n  Merge succeeded without GGUF export.")
            print(f"  Merged HF artifacts are available at: {MERGED_MODEL_DIR}")
            if gguf_error:
                print(f"  GGUF export issue: {gguf_error}")

        print(f"\n{'=' * 60}")
        print("  Merge complete")
        print(f"  Merged model: {MERGED_MODEL_DIR}")
        if gguf_exported:
            print(f"  GGUF output:  {GGUF_DIR}")
        else:
            print("  GGUF output:  skipped or failed softly")
        print(f"{'=' * 60}")
        return 0
    except Exception as exc:
        summary["merge_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if merge_error is not None and summary["cpu_fallback_used"]:
            summary["gpu_merge_error"] = "".join(
                traceback.format_exception_only(type(merge_error), merge_error)
            ).strip()
        raise
    finally:
        save_merge_summary(summary)
        release_model(model)
        if processor is not None:
            del processor
        if tokenizer is not None:
            del tokenizer
        if "model" in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("  Gemma4GR Phase 2 - Merge Adapters")
    print("=" * 60 + "\n")
    raise SystemExit(merge())
