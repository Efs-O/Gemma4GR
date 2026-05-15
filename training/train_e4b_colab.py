"""
Step 8 — Fine-tune Gemma 4 E4B for Greek STT (Google Colab).
Requires: L4 or A100 for practical VRAM headroom.
Uses the same multimodal dataset contract as the local E2B STT path.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent.parent
TRAIN_DATA = BASE / "data" / "train_stt.jsonl"
VAL_DATA = BASE / "data" / "train_stt_val.jsonl"
OUTPUT_DIR = BASE / "output" / "e4b_greek_stt"

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E4B_MODEL", "unsloth/gemma-4-E4B-it")

MAX_SEQ_LEN = 4096
LORA_R = 32
LORA_ALPHA = 64
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
WARMUP_STEPS = 5
LOGGING_STEPS = 1
EVAL_STEPS = 100
SAVE_STEPS = 500


def train():
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir.mkdir(exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"VRAM: {vram:.1f} GB")
    if vram < 16:
        print("[WARN] E4B usually needs an L4 or A100 class runtime.")

    print(f"\nLoading {MODEL_NAME} ...")
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=None,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
    )

    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    val_ds = load_dataset("json", data_files=str(VAL_DATA), split="train")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

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
            eval_steps=EVAL_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            fp16=not use_bf16,
            bf16=use_bf16,
            use_gradient_checkpointing="unsloth",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            max_length=None,
        ),
    )

    print("\nTraining E4B ...")
    stats = trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    gguf_dir = OUTPUT_DIR / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    model.save_pretrained_gguf(str(gguf_dir), processor.tokenizer, quantization_method="q4_k_m")

    with open(OUTPUT_DIR / "training_stats.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "train_loss": stats.training_loss}, f, indent=2)

    print(f"\nE4B training complete. Loss: {stats.training_loss:.4f}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  GGUF:    {gguf_dir}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Gemma4GR — Train E4B Greek STT (Colab)")
    print("=" * 55 + "\n")
    if not TRAIN_DATA.exists():
        print(f"[ERROR] {TRAIN_DATA} not found. Upload data to Colab first.")
        sys.exit(1)
    train()
