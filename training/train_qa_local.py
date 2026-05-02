"""
Phase 2 - Fine-tune Gemma 4 E2B for Greek text Q&A (local GPU).
Text-only LoRA, language layers only.
"""
import json
import os
import sys
from datetime import datetime
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

TRAIN_DATA = BASE / "data" / "train_qa.jsonl"
VAL_DATA = BASE / "data" / "val_qa.jsonl"
OUTPUT_DIR = BASE / "output" / "e2b_greek_qa"
LOG_DIR = BASE / "logs"
VALIDATION_STATUS = LOG_DIR / "validation_status.json"

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")
MODEL_PATH_OVERRIDE = os.getenv("E2B_MODEL_PATH", "").strip()

MAX_SEQ_LEN = int(os.getenv("QA_TRAIN_MAX_SEQ_LEN", "2048"))
LORA_R = int(os.getenv("QA_TRAIN_LORA_R", "32"))
LORA_ALPHA = int(os.getenv("QA_TRAIN_LORA_ALPHA", "64"))
EPOCHS = int(os.getenv("QA_TRAIN_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("QA_TRAIN_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("QA_TRAIN_GRAD_ACCUM", "4"))
LR = float(os.getenv("QA_TRAIN_LR", "1e-4"))
WARMUP_STEPS = int(os.getenv("QA_TRAIN_WARMUP_STEPS", "10"))
LOGGING_STEPS = int(os.getenv("QA_TRAIN_LOGGING_STEPS", "5"))
EVAL_STEPS = int(os.getenv("QA_TRAIN_EVAL_STEPS", "50"))
SAVE_STEPS = int(os.getenv("QA_TRAIN_SAVE_STEPS", "200"))
MAX_STEPS = int(os.getenv("QA_TRAIN_MAX_STEPS", "-1"))


def resolve_model_source() -> str:
    if MODEL_PATH_OVERRIDE:
        path = Path(MODEL_PATH_OVERRIDE)
        if path.exists():
            return str(path)

    default_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--gemma-4-E2B-it"
    refs_main = default_cache / "refs" / "main"
    snapshots = default_cache / "snapshots"
    if refs_main.exists() and snapshots.exists():
        snapshot = snapshots / refs_main.read_text(encoding="utf-8", errors="replace").strip()
        if snapshot.exists():
            return str(snapshot)

    return MODEL_NAME


def load_unsloth_model(FastModel):
    model_source = resolve_model_source()
    common_kwargs = dict(
        model_name=model_source,
        dtype=None,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
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


def warn_if_validation_stale(paths: list[Path], label: str) -> None:
    if not VALIDATION_STATUS.exists():
        print(f"  [WARN] No validation stamp found. Run step S (validate_samples.py) before {label}.")
        return

    try:
        status = json.loads(VALIDATION_STATUS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [WARN] Validation stamp is unreadable: {VALIDATION_STATUS}")
        return

    validated_files = status.get("files", {})
    stale = []
    for path in paths:
        rel = str(path.relative_to(BASE))
        if not path.exists():
            continue
        validated_mtime = validated_files.get(rel)
        if validated_mtime is None or path.stat().st_mtime > validated_mtime:
            stale.append(rel)

    if stale:
        print(f"  [WARN] Sample validation is stale for {label}. Re-run step S before a long training run.")
        for rel in stale:
            print(f"         Changed since validation: {rel}")


def check_prerequisites() -> int:
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Training data not found: {TRAIN_DATA}")
        print("  Run step E (prepare_qa_dataset.py) first.")
        sys.exit(1)

    with open(TRAIN_DATA, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  Training examples: {count}")
    warn_if_validation_stale([TRAIN_DATA, VAL_DATA], "QA training")

    if count < 50:
        print(f"  [WARN] Only {count} examples. Recommend 200+ for quality results.")

    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
        print(f"  HF Token: {HF_TOKEN[:8]}...")
    else:
        print("  [WARN] No HF_TOKEN; model download may fail for gated models")

    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 5:
            print("  [WARN] Less than 5 GB VRAM; reduce BATCH_SIZE to 1")

    return count


def train() -> None:
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    run_name = f"e2b_qa_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    print(f"\n[1/5] Loading {MODEL_NAME} (4-bit QLoRA) ...")
    print(f"  Source: {resolve_model_source()}")
    model, processor = load_unsloth_model(FastModel)
    print("  Model loaded")
    print(f"  Processing class: {type(processor).__name__}")

    print(f"\n[2/5] Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, language layers only) ...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
    )
    tokenizer = get_chat_template(processor.tokenizer, chat_template="gemma-4")
    processor.tokenizer = tokenizer

    print("\n[3/5] Loading Q&A dataset ...")
    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    val_ds = load_dataset("json", data_files=str(VAL_DATA), split="train")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    print("\n[4/5] Setting up trainer ...")
    use_bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(run_dir),
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            logging_steps=LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            max_steps=MAX_STEPS,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            fp16=not use_bf16,
            bf16=use_bf16,
            gradient_checkpointing=True,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            max_length=None,
        ),
    )

    print("\n[5/5] Training Greek Q&A LoRA ...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}")
    print(f"  LR: {LR} | LoRA r: {LORA_R} | Max steps: {MAX_STEPS}\n")

    stats = trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    processor.tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n  QA LoRA adapter saved: {adapter_dir}")

    stats_path = OUTPUT_DIR / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "epochs": EPOCHS,
                "lora_r": LORA_R,
                "max_steps": MAX_STEPS,
                "train_loss": stats.training_loss,
                "run_dir": str(run_dir),
                "phase": "qa",
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Final loss:   {stats.training_loss:.4f}")
    print(f"  Adapter:      {adapter_dir}")
    print("  Next step:    Run step G to merge STT + QA adapters -> GGUF")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Gemma4GR Phase 2 - Train Greek Q&A LoRA (Local)")
    print("=" * 60 + "\n")
    count = check_prerequisites()
    if count == 0:
        sys.exit(1)
    train()
