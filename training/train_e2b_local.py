"""
Step 7 — Fine-tune Gemma 4 E2B for Greek STT (local GPU).
Target GPU: RTX 4060 Ti / 5060 Ti 16 GB
VRAM usage: ~8-10 GB with QLoRA 4-bit
Est. time: 2-4 hours for 3000 pairs x 3 epochs
"""
import csv
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

from env_bootstrap import ensure_unsloth_runtime, normalize_hf_model_path, resolve_hf_snapshot

ensure_unsloth_runtime(BASE)

import torch
from dotenv import load_dotenv

load_dotenv()
_train_data_raw = os.getenv("STT_TRAIN_DATA", "").strip()
TRAIN_DATA = Path(_train_data_raw) if _train_data_raw else BASE / "data" / "train_stt.jsonl"
if not TRAIN_DATA.is_absolute():
    TRAIN_DATA = BASE / TRAIN_DATA
_val_data_raw = os.getenv("STT_VAL_DATA", "").strip()
VAL_DATA = Path(_val_data_raw) if _val_data_raw else BASE / "data" / "train_stt_val.jsonl"
if not VAL_DATA.is_absolute():
    VAL_DATA = BASE / VAL_DATA
OUTPUT_DIR = BASE / "output" / "e2b_greek_stt"
LOG_DIR = BASE / "logs"
VALIDATION_STATUS = LOG_DIR / "validation_status.json"

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")
MODEL_PATH_OVERRIDE = os.getenv("E2B_MODEL_PATH", "").strip()

MAX_SEQ_LEN = int(os.getenv("STT_TRAIN_MAX_SEQ_LEN", "1024"))
LORA_R = int(os.getenv("STT_TRAIN_LORA_R", "64"))
LORA_ALPHA = int(os.getenv("STT_TRAIN_LORA_ALPHA", "128"))
EPOCHS = int(os.getenv("STT_TRAIN_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("STT_TRAIN_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("STT_TRAIN_GRAD_ACCUM", "4"))
LR = float(os.getenv("STT_TRAIN_LR", "2e-4"))
WARMUP_STEPS = int(os.getenv("STT_TRAIN_WARMUP_STEPS", "5"))
LOGGING_STEPS = int(os.getenv("STT_TRAIN_LOGGING_STEPS", "1"))
EVAL_STRATEGY = os.getenv("STT_TRAIN_EVAL_STRATEGY", "epoch").strip().lower() or "epoch"
EVAL_STEPS = int(os.getenv("STT_TRAIN_EVAL_STEPS", "594"))
SAVE_STRATEGY = os.getenv("STT_TRAIN_SAVE_STRATEGY", EVAL_STRATEGY).strip().lower() or EVAL_STRATEGY
SAVE_STEPS = int(os.getenv("STT_TRAIN_SAVE_STEPS", "594"))
SAVE_TOTAL_LIMIT = int(os.getenv("STT_TRAIN_SAVE_TOTAL_LIMIT", "2"))
MAX_STEPS = int(os.getenv("STT_TRAIN_MAX_STEPS", "-1"))
DATASET_NUM_PROC = int(os.getenv("STT_TRAIN_DATASET_NUM_PROC", "1"))
LOAD_BEST_MODEL_AT_END = os.getenv("STT_TRAIN_LOAD_BEST_MODEL_AT_END", "1").strip().lower() in ("1", "true", "yes")


def save_training_metrics(trainer, run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    log_history = trainer.state.log_history
    json_path = output_dir / "metrics_history.json"
    csv_path = output_dir / "metrics_history.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    fieldnames: list[str] = []
    for row in log_history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_history:
            writer.writerow(row)

    run_json = run_dir / "metrics_history.json"
    run_csv = run_dir / "metrics_history.csv"
    run_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    run_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, csv_path


def resolve_model_source() -> str:
    if MODEL_PATH_OVERRIDE:
        normalized = normalize_hf_model_path(MODEL_PATH_OVERRIDE)
        if normalized:
            return normalized

    snapshot = resolve_hf_snapshot("models--unsloth--gemma-4-E2B-it")
    if snapshot:
        return snapshot

    return MODEL_NAME


def load_unsloth_model(FastModel):
    common_kwargs = dict(
        model_name=resolve_model_source(),
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


def get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def warn_if_validation_stale(paths: list[Path], label: str):
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
        print("  Run step 4 (prepare_stt_dataset.py) first.")
        sys.exit(1)
    if not VAL_DATA.exists():
        print(f"[ERROR] Validation data not found: {VAL_DATA}")
        print("  Run step 4 (prepare_stt_dataset.py) first.")
        sys.exit(1)

    with open(TRAIN_DATA, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  Training examples: {count}")
    warn_if_validation_stale([TRAIN_DATA, VAL_DATA], "STT training")

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
            print("  [WARN] Less than 7 GB VRAM — reduce batch size or sequence length")

    return count


def train():
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    run_name = f"e2b_greek_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    print(f"\n[1/5] Loading {MODEL_NAME} (4-bit QLoRA) ...")
    print(f"  Source: {resolve_model_source()}")
    model, processor = load_unsloth_model(FastModel)
    tokenizer = get_tokenizer(processor)
    print("  Model loaded")

    print(f"\n[2/5] Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    # Gemma 4 routes audio through the multimodal vision/audio stack.
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

    print(f"\n[3/5] Loading dataset ...")
    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    val_ds = load_dataset("json", data_files=str(VAL_DATA), split="train")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    print(f"\n[4/5] Setting up trainer ...")
    use_bf16 = torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(run_dir),
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            logging_steps=LOGGING_STEPS,
            eval_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            save_strategy=SAVE_STRATEGY,
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            max_steps=MAX_STEPS,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            fp16=not use_bf16,
            bf16=use_bf16,
            gradient_checkpointing=True,
            report_to="none",
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataset_num_proc=DATASET_NUM_PROC,
            max_length=None,
        ),
    )

    print(f"\n[5/5] Training ...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}")
    print(
        f"  LR: {LR} | LoRA r: {LORA_R} | Max seq len: {MAX_SEQ_LEN} | "
        f"Eval: {EVAL_STRATEGY}/{EVAL_STEPS} | Save: {SAVE_STRATEGY}/{SAVE_STEPS} | "
        f"Best-at-end: {LOAD_BEST_MODEL_AT_END} | Dataset proc: {DATASET_NUM_PROC}\n"
    )

    stats = trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\n  LoRA adapter saved: {adapter_dir}")

    metrics_json, metrics_csv = save_training_metrics(trainer, run_dir, OUTPUT_DIR)
    print(f"  Metrics JSON: {metrics_json}")
    print(f"  Metrics CSV:  {metrics_csv}")

    gguf_dir = OUTPUT_DIR / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    print("  Exporting GGUF (q4_k_m) ...")
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"  GGUF saved: {gguf_dir}")

    stats_path = OUTPUT_DIR / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "epochs": EPOCHS,
                "lora_r": LORA_R,
                "max_seq_len": MAX_SEQ_LEN,
                "max_steps": MAX_STEPS,
                "train_loss": stats.training_loss,
                "run_dir": str(run_dir),
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 55}")
    print("  Training complete!")
    print(f"  Final loss:   {stats.training_loss:.4f}")
    print(f"  Adapter:      {adapter_dir}")
    print(f"  GGUF:         {gguf_dir}")
    print(f"  Metrics:      {metrics_csv}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Gemma4GR — Train E2B Greek STT (Local)")
    print("=" * 55 + "\n")
    count = check_prerequisites()
    if count == 0:
        sys.exit(1)
    train()
