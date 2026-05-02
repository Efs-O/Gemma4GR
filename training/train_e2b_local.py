"""
Step 7 — Fine-tune Gemma 4 E2B for Greek STT (local GPU).
Target GPU: RTX 4060 Ti / 5060 Ti 16 GB
VRAM usage: ~8-10 GB with QLoRA 4-bit
Est. time:  2-4 hours for 3000 pairs × 3 epochs
"""
import os, sys, json, torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE       = Path(__file__).parent.parent
TRAIN_DATA = BASE / "data" / "train_stt.jsonl"
VAL_DATA   = BASE / "data" / "train_stt_val.jsonl"
OUTPUT_DIR = BASE / "output" / "e2b_greek_stt"
LOG_DIR    = BASE / "logs"

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")

# ── Hyperparameters ──────────────────────────────────────────────────────
MAX_SEQ_LEN   = 4096
LORA_R        = 64
LORA_ALPHA    = 128
EPOCHS        = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 4        # Effective batch = 4
LR            = 2e-4
WARMUP_STEPS  = 5
LOGGING_STEPS = 1
EVAL_STEPS    = 100
SAVE_STEPS    = 500


def check_prerequisites():
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Training data not found: {TRAIN_DATA}")
        print("  Run step 4 (prepare_stt_dataset.py) first.")
        sys.exit(1)

    with open(TRAIN_DATA) as f:
        count = sum(1 for _ in f)
    print(f"  Training examples: {count}")

    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
        print(f"  HF Token: {HF_TOKEN[:8]}...✓")
    else:
        print("  [WARN] No HF_TOKEN — model download may fail for gated models")

    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 7:
            print("  [WARN] Less than 7 GB VRAM — reduce BATCH_SIZE or MAX_SEQ_LEN")

    return count


def train():
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    run_name  = f"e2b_greek_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir   = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {MODEL_NAME} (4-bit QLoRA) ...")
    model, processor = FastModel.from_pretrained(
        model_name    = MODEL_NAME,
        dtype         = None,
        max_seq_length= MAX_SEQ_LEN,
        load_in_4bit  = True,
        full_finetuning=False,
    )
    print("  Model loaded ✓")

    # ── Apply LoRA ────────────────────────────────────────────────────────
    print(f"\n[2/5] Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers    = False,
        finetune_language_layers  = True,
        finetune_attention_modules= True,
        finetune_mlp_modules      = True,
        r           = LORA_R,
        lora_alpha  = LORA_ALPHA,
        lora_dropout= 0,
        bias        = "none",
    )

    # ── Chat template ─────────────────────────────────────────────────────
    tokenizer = get_chat_template(processor.tokenizer, chat_template="gemma-4")
    processor.tokenizer = tokenizer

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading dataset ...")
    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    val_ds   = load_dataset("json", data_files=str(VAL_DATA),   split="train")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Trainer ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Setting up trainer ...")
    use_bf16 = torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = processor.tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args = SFTConfig(
            output_dir                  = str(run_dir),
            dataset_text_field          = "text",
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps                = WARMUP_STEPS,
            num_train_epochs            = EPOCHS,
            learning_rate               = LR,
            logging_steps               = LOGGING_STEPS,
            eval_steps                  = EVAL_STEPS,
            save_steps                  = SAVE_STEPS,
            save_total_limit            = 3,
            optim                       = "adamw_8bit",
            weight_decay                = 0.001,
            lr_scheduler_type           = "linear",
            fp16                        = not use_bf16,
            bf16                        = use_bf16,
            use_gradient_checkpointing  = "unsloth",
            report_to                   = "none",
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            greater_is_better           = False,
        ),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training ...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}")
    print(f"  LR: {LR} | LoRA r: {LORA_R}\n")

    stats = trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n  LoRA adapter saved: {adapter_dir}")

    # Export GGUF
    gguf_dir = OUTPUT_DIR / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    print(f"  Exporting GGUF (q4_k_m) ...")
    model.save_pretrained_gguf(
        str(gguf_dir),
        processor.tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"  GGUF saved: {gguf_dir}")

    # Save training stats
    stats_path = OUTPUT_DIR / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "model":      MODEL_NAME,
            "epochs":     EPOCHS,
            "lora_r":     LORA_R,
            "train_loss": stats.training_loss,
            "run_dir":    str(run_dir),
        }, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Final loss:   {stats.training_loss:.4f}")
    print(f"  Adapter:      {adapter_dir}")
    print(f"  GGUF:         {gguf_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Gemma4GR — Train E2B Greek STT (Local)")
    print("=" * 55 + "\n")
    count = check_prerequisites()
    if count == 0:
        sys.exit(1)
    train()
