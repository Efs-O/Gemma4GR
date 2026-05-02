"""
Phase 2 — Step G: Merge STT LoRA + QA LoRA into base model → export GGUF.

Merge order:
  1. Load base model (float16, not quantized)
  2. Apply + merge STT adapter (output/e2b_greek_stt/lora_adapter)
  3. Apply + merge QA adapter  (output/e2b_greek_qa/lora_adapter)
  4. Export GGUF q4_k_m + q8_0 → output/merged_gguf/

The merged GGUF is drop-in compatible with Ollama (ollama create) and llama.cpp.
"""
import os, sys, json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE        = Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from env_bootstrap import ensure_unsloth_runtime

ensure_unsloth_runtime(BASE)

import torch
from dotenv import load_dotenv

load_dotenv()

STT_ADAPTER = BASE / "output" / "e2b_greek_stt" / "lora_adapter"
QA_ADAPTER  = BASE / "output" / "e2b_greek_qa"  / "lora_adapter"
OUTPUT_DIR  = BASE / "output" / "merged_gguf"

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")


def load_unsloth_model(FastModel, dtype):
    common_kwargs = dict(
        model_name=MODEL_NAME,
        dtype=dtype,
        max_seq_length=4096,
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


def check_prerequisites() -> dict:
    status = {
        "stt_adapter": STT_ADAPTER.exists(),
        "qa_adapter":  QA_ADAPTER.exists(),
    }

    print("  Adapter status:")
    print(f"    STT adapter: {'✓ Found' if status['stt_adapter'] else '✗ Missing'} — {STT_ADAPTER}")
    print(f"    QA  adapter: {'✓ Found' if status['qa_adapter']  else '✗ Missing'} — {QA_ADAPTER}")

    if not status["stt_adapter"] and not status["qa_adapter"]:
        print("\n[ERROR] No adapters found. Run steps 7 and F first.")
        sys.exit(1)

    if not status["stt_adapter"]:
        print("\n[WARN] STT adapter missing — will merge QA only.")
    if not status["qa_adapter"]:
        print("\n[WARN] QA adapter missing — will merge STT only.")

    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 10:
            print("  [WARN] Merging in float16 needs ~10 GB VRAM. Consider running on CPU if OOM.")
    else:
        print("  [INFO] No CUDA — merge will run on CPU (slow but works).")

    return status


def merge():
    from unsloth import FastModel
    from peft import PeftModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    status = check_prerequisites()

    # ── Load base in float16 (not quantized — needed for clean merge) ─────────
    print(f"\n[1/4] Loading base model {MODEL_NAME} in float16 ...")
    print("  (This may take a few minutes and use ~8 GB RAM/VRAM)\n")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model, processor = load_unsloth_model(FastModel, dtype)
    tokenizer = get_tokenizer(processor)
    print("  Base model loaded ✓")

    # ── Merge STT adapter ─────────────────────────────────────────────────────
    if status["stt_adapter"]:
        print(f"\n[2/4] Merging STT LoRA adapter ...")
        model = PeftModel.from_pretrained(model, str(STT_ADAPTER))
        model = model.merge_and_unload()
        print("  STT adapter merged ✓")
    else:
        print("\n[2/4] Skipping STT adapter (not found)")

    # ── Merge QA adapter ──────────────────────────────────────────────────────
    if status["qa_adapter"]:
        print(f"\n[3/4] Merging QA LoRA adapter ...")
        model = PeftModel.from_pretrained(model, str(QA_ADAPTER))
        model = model.merge_and_unload()
        print("  QA adapter merged ✓")
    else:
        print("\n[3/4] Skipping QA adapter (not found)")

    # ── Export GGUF ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Exporting GGUF to {OUTPUT_DIR} ...")
    print("  Quantization: q4_k_m (primary) + q8_0 (high quality)\n")

    model.save_pretrained_gguf(
        str(OUTPUT_DIR),
        tokenizer,
        quantization_method=["q4_k_m", "q8_0"],
    )

    # ── Write Ollama Modelfile ─────────────────────────────────────────────────
    gguf_files = list(OUTPUT_DIR.glob("*.gguf"))
    q4_file = next((f for f in gguf_files if "q4" in f.name.lower()), None)

    if q4_file:
        modelfile_path = OUTPUT_DIR / "Modelfile"
        modelfile_path.write_text(
            f'FROM ./{q4_file.name}\n\n'
            'SYSTEM """\nΕίσαι ένας βοηθός που μιλά άπταιστα ελληνικά. '
            'Απαντάς πάντα στα ελληνικά, με φυσική και σωστή γλώσσα.\n"""\n\n'
            'PARAMETER temperature 0.7\n'
            'PARAMETER num_ctx 4096\n',
            encoding="utf-8"
        )
        print(f"\n  Modelfile written: {modelfile_path}")
        print(f"\n  To load in Ollama:")
        print(f"    cd {OUTPUT_DIR}")
        print(f"    ollama create gemma4gr -f Modelfile")
        print(f"    ollama run gemma4gr")

    # ── Save merge summary ────────────────────────────────────────────────────
    summary = {
        "base_model":  MODEL_NAME,
        "stt_merged":  status["stt_adapter"],
        "qa_merged":   status["qa_adapter"],
        "output_dir":  str(OUTPUT_DIR),
        "gguf_files":  [f.name for f in gguf_files],
    }
    (OUTPUT_DIR / "merge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"  Merge complete!")
    print(f"  GGUF output: {OUTPUT_DIR}")
    for f in gguf_files:
        size_gb = f.stat().st_size / 1e9
        print(f"    {f.name} ({size_gb:.1f} GB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Gemma4GR Phase 2 — Merge Adapters → GGUF")
    print("=" * 60 + "\n")
    merge()
