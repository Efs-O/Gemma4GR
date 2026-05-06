"""
Track C Step 3 — Fine-tune Gemma 4 E2B on audio Q&A pairs (local GPU).

Input:  data/train_stt_qa.jsonl by default, or STT_QA_TRAIN_DATA
        data/val_stt_qa.jsonl by default, or STT_QA_VAL_DATA
Output: output/e2b_stt_qa/lora_adapter

Trains Gemma to answer Greek questions spoken in the JOY voice.
Audio format: 16 kHz mono WAV. Language + vision layers on (audio goes through
the audio encoder, which is part of the vision stack in Gemma 4).

Run:
  python training/train_stt_qa_local.py
"""
from __future__ import annotations

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

from env_bootstrap import ensure_unsloth_runtime, resolve_hf_snapshot

ensure_unsloth_runtime(BASE)

import torch
from dotenv import load_dotenv

try:
    import psutil
except Exception:
    psutil = None

load_dotenv()

_train_data_raw = os.getenv("STT_QA_TRAIN_DATA", "").strip()
TRAIN_DATA = Path(_train_data_raw) if _train_data_raw else BASE / "data" / "train_stt_qa.jsonl"
if not TRAIN_DATA.is_absolute():
    TRAIN_DATA = BASE / TRAIN_DATA
_val_data_raw = os.getenv("STT_QA_VAL_DATA", "").strip()
VAL_DATA = Path(_val_data_raw) if _val_data_raw else BASE / "data" / "val_stt_qa.jsonl"
if not VAL_DATA.is_absolute():
    VAL_DATA = BASE / VAL_DATA
OUTPUT_DIR_NAME = os.getenv("STT_QA_OUTPUT_DIR", "").strip() or "e2b_stt_qa"
OUTPUT_DIR = BASE / "output" / OUTPUT_DIR_NAME

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("STT_QA_MODEL", "").strip() or os.getenv("E2B_MODEL", "unsloth/gemma-4-E2B-it")
MODEL_PATH_OVERRIDE = os.getenv("STT_QA_MODEL_PATH", "").strip() or os.getenv("E2B_MODEL_PATH", "").strip()

MAX_SEQ_LEN = int(os.getenv("STT_QA_MAX_SEQ_LEN", "1024"))
LORA_R = int(os.getenv("STT_QA_LORA_R", "64"))
LORA_ALPHA = int(os.getenv("STT_QA_LORA_ALPHA", "128"))
EPOCHS = int(os.getenv("STT_QA_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("STT_QA_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("STT_QA_GRAD_ACCUM", "4"))
LR = float(os.getenv("STT_QA_LR", "2e-4"))
WARMUP_STEPS = int(os.getenv("STT_QA_WARMUP_STEPS", "5"))
LOGGING_STEPS = int(os.getenv("STT_QA_LOGGING_STEPS", "1"))
EVAL_STRATEGY = os.getenv("STT_QA_EVAL_STRATEGY", "epoch").strip().lower() or "epoch"
EVAL_STEPS = int(os.getenv("STT_QA_EVAL_STEPS", "594"))
SAVE_STRATEGY = os.getenv("STT_QA_SAVE_STRATEGY", EVAL_STRATEGY).strip().lower() or EVAL_STRATEGY
SAVE_STEPS = int(os.getenv("STT_QA_SAVE_STEPS", "594"))
SAVE_TOTAL_LIMIT = int(os.getenv("STT_QA_SAVE_TOTAL_LIMIT", "2"))
MAX_STEPS = int(os.getenv("STT_QA_MAX_STEPS", "-1"))
_dataset_num_proc_raw = os.getenv("STT_QA_DATASET_NUM_PROC", "").strip()
DATASET_NUM_PROC = int(_dataset_num_proc_raw) if _dataset_num_proc_raw else None
LOAD_BEST_MODEL_AT_END = os.getenv("STT_QA_LOAD_BEST_MODEL_AT_END", "1").strip().lower() in ("1", "true", "yes")


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


class MemoryMetricsLogger:
    def __init__(self, run_dir: Path, output_dir: Path):
        self.run_dir = run_dir
        self.output_dir = output_dir
        self.jsonl_path = run_dir / "memory_metrics.jsonl"
        self.summary_path = output_dir / "memory_summary.json"
        self.process = psutil.Process() if psutil is not None else None
        self.max_process_rss_gb = 0.0
        self.max_system_used_gb = 0.0

    @staticmethod
    def _bytes_to_gb(value: int | float) -> float:
        return round(float(value) / (1024 ** 3), 4)

    def snapshot(
        self,
        stage: str,
        step: int | None = None,
        epoch: float | None = None,
        extra: dict | None = None,
    ) -> dict:
        row: dict[str, object] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "step": step,
            "epoch": round(epoch, 4) if isinstance(epoch, (int, float)) else epoch,
        }

        if self.process is not None:
            rss_gb = self._bytes_to_gb(self.process.memory_info().rss)
            self.max_process_rss_gb = max(self.max_process_rss_gb, rss_gb)
            row["process_rss_gb"] = rss_gb

        if psutil is not None:
            vm = psutil.virtual_memory()
            system_used_gb = self._bytes_to_gb(vm.total - vm.available)
            self.max_system_used_gb = max(self.max_system_used_gb, system_used_gb)
            row["system_used_gb"] = system_used_gb
            row["system_available_gb"] = self._bytes_to_gb(vm.available)
            row["system_used_percent"] = round(vm.percent, 1)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            max_allocated = torch.cuda.max_memory_allocated(0)
            max_reserved = torch.cuda.max_memory_reserved(0)
            row["gpu_allocated_gb"] = self._bytes_to_gb(allocated)
            row["gpu_reserved_gb"] = self._bytes_to_gb(reserved)
            row["gpu_max_allocated_gb"] = self._bytes_to_gb(max_allocated)
            row["gpu_max_reserved_gb"] = self._bytes_to_gb(max_reserved)

        if extra:
            row.update(extra)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def write_summary(self, trainer_state=None) -> Path:
        summary: dict[str, object] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "max_process_rss_gb": round(self.max_process_rss_gb, 4),
            "max_system_used_gb": round(self.max_system_used_gb, 4),
        }
        if torch.cuda.is_available():
            summary["gpu_max_allocated_gb"] = self._bytes_to_gb(torch.cuda.max_memory_allocated(0))
            summary["gpu_max_reserved_gb"] = self._bytes_to_gb(torch.cuda.max_memory_reserved(0))
        if trainer_state is not None:
            summary["global_step"] = getattr(trainer_state, "global_step", None)
            summary["epoch"] = getattr(trainer_state, "epoch", None)

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        (self.run_dir / "memory_summary.json").write_text(
            self.summary_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return self.summary_path


def resolve_model_source() -> str:
    if MODEL_PATH_OVERRIDE:
        path = Path(MODEL_PATH_OVERRIDE)
        if path.exists():
            return str(path)

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
        print("  [INFO] Retrying from local cache only ...")
        return FastModel.from_pretrained(**common_kwargs, local_files_only=True)


def check_prerequisites() -> int:
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Training data not found: {TRAIN_DATA}")
        print("  Run prepare_stt_qa_dataset.py first.")
        sys.exit(1)
    if not VAL_DATA.exists():
        print(f"[ERROR] Validation data not found: {VAL_DATA}")
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
    from transformers import Trainer, TrainerCallback
    from unsloth import FastModel

    class AudioSafeSFTTrainer(SFTTrainer):
        """Bypass TRL metric code paths that break on Gemma audio outputs."""

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            inputs["use_cache"] = False
            return Trainer.compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    class MemorySnapshotCallback(TrainerCallback):
        def __init__(self, logger: MemoryMetricsLogger):
            self.logger = logger

        def on_train_begin(self, args, state, control, **kwargs):
            self.logger.snapshot("train_begin", step=state.global_step, epoch=state.epoch)
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.logger.snapshot("epoch_begin", step=state.global_step, epoch=state.epoch)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step and state.global_step % 25 == 0:
                self.logger.snapshot("step_end", step=state.global_step, epoch=state.epoch)
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.logger.snapshot(
                "evaluate",
                step=state.global_step,
                epoch=state.epoch,
                extra={"eval_loss": metrics.get("eval_loss") if metrics else None},
            )
            return control

        def on_save(self, args, state, control, **kwargs):
            self.logger.snapshot("save", step=state.global_step, epoch=state.epoch)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self.logger.snapshot("epoch_end", step=state.global_step, epoch=state.epoch)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.logger.snapshot("train_end", step=state.global_step, epoch=state.epoch)
            self.logger.write_summary(state)
            return control

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_tag = "e4b" if "E4B" in MODEL_NAME.upper() else "e2b"
    run_name = f"{model_tag}_stt_qa_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)
    memory_logger = MemoryMetricsLogger(run_dir, OUTPUT_DIR)

    print(f"\n[1/5] Loading {MODEL_NAME} (4-bit QLoRA) ...")
    print(f"  Source: {resolve_model_source()}")
    model, processor = load_unsloth_model(FastModel)
    print("  Model loaded")
    memory_logger.snapshot("model_loaded")

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
    memory_logger.snapshot(
        "dataset_loaded",
        extra={"train_examples": len(train_ds), "val_examples": len(val_ds)},
    )

    print("\n[4/5] Setting up trainer ...")
    use_bf16 = torch.cuda.is_bf16_supported()

    trainer = AudioSafeSFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[MemorySnapshotCallback(memory_logger)],
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

    print("\n[5/5] Training audio Q&A LoRA ...")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}")
    print(
        f"  LR: {LR} | LoRA r: {LORA_R} | Max seq len: {MAX_SEQ_LEN} | "
        f"Eval: {EVAL_STRATEGY}/{EVAL_STEPS} | Save: {SAVE_STRATEGY}/{SAVE_STEPS} | "
        f"Best-at-end: {LOAD_BEST_MODEL_AT_END} | Dataset proc: {DATASET_NUM_PROC} | "
        f"Max steps: {MAX_STEPS}\n"
    )

    stats = trainer.train()
    memory_logger.snapshot("post_train", step=trainer.state.global_step, epoch=trainer.state.epoch)

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\n  LoRA adapter saved: {adapter_dir}")
    memory_logger.snapshot("adapter_saved", step=trainer.state.global_step, epoch=trainer.state.epoch)

    metrics_json, metrics_csv = save_training_metrics(trainer, run_dir, OUTPUT_DIR)
    print(f"  Metrics JSON: {metrics_json}")
    print(f"  Metrics CSV:  {metrics_csv}")
    memory_summary = memory_logger.write_summary(trainer.state)
    print(f"  Memory log:   {memory_logger.jsonl_path}")
    print(f"  Memory sum:   {memory_summary}")

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
                "phase": "stt_qa",
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Final loss:   {stats.training_loss:.4f}")
    print(f"  Adapter:      {adapter_dir}")
    print(f"  Metrics:      {metrics_csv}")
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
