"""
Track C Step 3 — Fine-tune Gemma 4 E2B on audio Q&A pairs (local GPU).

Input:  data/train_stt_qa.jsonl  (from prepare_stt_qa_dataset.py)
        data/val_stt_qa.jsonl
Output: output/e2b_stt_qa/lora_adapter

Trains Gemma to answer Greek questions spoken in the JOY voice.
Audio format: 16 kHz mono WAV. Language + vision layers on (audio goes through
the audio encoder, which is part of the vision stack in Gemma 4).

Run:
  python training/train_stt_qa_local.py
"""
from __future__ import annotations

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

TRAIN_DATA = BASE / "data" / "train_stt_qa.jsonl"
VAL_DATA = BASE / "data" / "val_stt_qa.jsonl"
OUTPUT_DIR = BASE / "output" / "e2b_stt_qa"

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")

MAX_SEQ_LEN = int(os.getenv("STT_QA_MAX_SEQ_LEN", "4096"))
LORA_R = int(os.getenv("STT_QA_LORA_R", "64"))
LORA_ALPHA = int(os.getenv("STT_QA_LORA_ALPHA", "128"))
EPOCHS = int(os.getenv("STT_QA_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("STT_QA_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("STT_QA_GRAD_ACCUM", "4"))
LR = float(os.getenv("STT_QA_LR", "2e-4"))
WARMUP_STEPS = int(os.getenv("STT_QA_WARMUP_STEPS", "5"))
LOGGING_STEPS = int(os.getenv("STT_QA_LOGGING_STEPS", "1"))
EVAL_STEPS = int(os.getenv("STT_QA_EVAL_STEPS", "100"))
SAVE_STEPS = int(os.getenv("STT_QA_SAVE_STEPS", "500"))


def load_unsloth_model(FastModel):
    common_kwargs = dict(
        model_name=MODEL_NAME,
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
        print("  [INFO] Retrying from local cache only ...")
        return FastModel.from_pretrained(**common_kwargs, local_files_only=True)


def check_prerequisites() -> int:
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Training data not found: {TRAIN_DATA}")
        print("  Run prepare_stt_qa_dataset.py first.")
        sys.exit(1)

    with open(TRAIN_DATA, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  Training examples: {count}")

    if count < 20:
        print(f"  [WARN] Only {count} examples — recommend 200+ for meaningful results.")

    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
        print(f"  HF Token: {HF_TOKEN[:8]}...")
    else:
        print("  [WARN] No HF_TOKEN — model download may fail for gated models")

    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 7:
            print("  [WARN] Less than 7 GB VRAM — reduce BATCH_SIZE or SEQ_LEN")

    return count


def train() -> None:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_name = f"e2b_stt_qa_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    print(f"\n[1/5] Loading {MODEL_NAME} (4-bit QLoRA) ...")
    model, processor = load_unsloth_model(FastModel)
    print("  Model loaded")

    print(f"\n[2/5] Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    # Vision layers ON — Gemma 4 routes audio through the vision/audio encoder
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
    )

    print("\n[3/5] Loading audio Q&A dataset ...")
    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    val_ds = load_dataset("json", data_files=str(VAL_DATA), split="train")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    print("\n[4/5] Setting up trainer ...")
    use_bf16 = torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(run_dir),
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
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
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

    print("\n[5/5] Training audio Q&A LoRA ...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}")
    print(f"  LR: {LR} | LoRA r: {LORA_R}\n")

    stats = trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\n  LoRA adapter saved: {adapter_dir}")

    stats_path = OUTPUT_DIR / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "epochs": EPOCHS,
                "lora_r": LORA_R,
                "train_loss": stats.training_loss,
                "run_dir": str(run_dir),
                "phase": "stt_qa",
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Final loss:   {stats.training_loss:.4f}")
    print(f"  Adapter:      {adapter_dir}")
    print("  Next step:    Add this adapter to merge_adapters.py for final GGUF")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Gemma4GR Track C — Train STT Q&A LoRA (Local)")
    print("=" * 60 + "\n")
    count = check_prerequisites()
    if count == 0:
        sys.exit(1)
    train()
