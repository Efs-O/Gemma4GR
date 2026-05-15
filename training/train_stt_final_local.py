"""
Train a strict STT-only Greek audio adapter from the final human-voice dataset.
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

from env_bootstrap import ensure_unsloth_runtime, normalize_hf_model_path, resolve_hf_snapshot

ensure_unsloth_runtime(BASE)

import torch
from dotenv import load_dotenv

try:
    import psutil
except Exception:
    psutil = None

load_dotenv()


def _resolve_output_dir(env_name: str, default_name: str) -> Path:
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else BASE / path
    root = os.getenv("GEMMA4GR_OUTPUT_ROOT", "").strip()
    if root:
        root_path = Path(root)
        root_path = root_path if root_path.is_absolute() else BASE / root_path
        return root_path / default_name
    return BASE / "output" / default_name


_train_data_raw = os.getenv("STT_FINAL_TRAIN_DATA", "").strip()
TRAIN_DATA = Path(_train_data_raw) if _train_data_raw else BASE / "data" / "train_stt_final.jsonl"
if not TRAIN_DATA.is_absolute():
    TRAIN_DATA = BASE / TRAIN_DATA
_val_data_raw = os.getenv("STT_FINAL_VAL_DATA", "").strip()
VAL_DATA = Path(_val_data_raw) if _val_data_raw else BASE / "data" / "val_stt_final.jsonl"
if not VAL_DATA.is_absolute():
    VAL_DATA = BASE / VAL_DATA
OUTPUT_DIR = _resolve_output_dir("STT_FINAL_OUTPUT_DIR", "e4b_stt_final")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("STT_FINAL_MODEL", "").strip() or "unsloth/gemma-4-E4B-it"
MODEL_PATH_OVERRIDE = os.getenv("STT_FINAL_MODEL_PATH", "").strip()

MAX_SEQ_LEN = int(os.getenv("STT_FINAL_MAX_SEQ_LEN", "1024"))
LORA_R = int(os.getenv("STT_FINAL_LORA_R", "32"))
LORA_ALPHA = int(os.getenv("STT_FINAL_LORA_ALPHA", "64"))
EPOCHS = int(os.getenv("STT_FINAL_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("STT_FINAL_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("STT_FINAL_GRAD_ACCUM", "4"))
LR = float(os.getenv("STT_FINAL_LR", "1e-4"))
WARMUP_STEPS = int(os.getenv("STT_FINAL_WARMUP_STEPS", "10"))
LOGGING_STEPS = int(os.getenv("STT_FINAL_LOGGING_STEPS", "5"))
EVAL_STRATEGY = os.getenv("STT_FINAL_EVAL_STRATEGY", "epoch").strip().lower() or "epoch"
EVAL_STEPS = int(os.getenv("STT_FINAL_EVAL_STEPS", "200"))
SAVE_STRATEGY = os.getenv("STT_FINAL_SAVE_STRATEGY", EVAL_STRATEGY).strip().lower() or EVAL_STRATEGY
SAVE_STEPS = int(os.getenv("STT_FINAL_SAVE_STEPS", "200"))
SAVE_TOTAL_LIMIT = int(os.getenv("STT_FINAL_SAVE_TOTAL_LIMIT", "3"))
MAX_STEPS = int(os.getenv("STT_FINAL_MAX_STEPS", "-1"))
_dataset_num_proc_raw = os.getenv("STT_FINAL_DATASET_NUM_PROC", "").strip()
DATASET_NUM_PROC = int(_dataset_num_proc_raw) if _dataset_num_proc_raw else None
LOAD_BEST_MODEL_AT_END = os.getenv("STT_FINAL_LOAD_BEST_MODEL_AT_END", "1").strip().lower() in ("1", "true", "yes")
FINETUNE_VISION_LAYERS = os.getenv("STT_FINAL_FINETUNE_VISION_LAYERS", "0").strip().lower() in ("1", "true", "yes")


def save_training_metrics(trainer, run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    log_history = trainer.state.log_history
    json_path = output_dir / "metrics_history.json"
    csv_path = output_dir / "metrics_history.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(log_history, handle, indent=2, ensure_ascii=False)
    fieldnames: list[str] = []
    for row in log_history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_history:
            writer.writerow(row)
    (run_dir / "metrics_history.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "metrics_history.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
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

    def snapshot(self, stage: str, step: int | None = None, epoch: float | None = None, extra: dict | None = None) -> dict:
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
            used_gb = self._bytes_to_gb(vm.total - vm.available)
            self.max_system_used_gb = max(self.max_system_used_gb, used_gb)
            row["system_used_gb"] = used_gb
        if torch.cuda.is_available():
            row["gpu_max_allocated_gb"] = self._bytes_to_gb(torch.cuda.max_memory_allocated(0))
            row["gpu_max_reserved_gb"] = self._bytes_to_gb(torch.cuda.max_memory_reserved(0))
        if extra:
            row.update(extra)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def write_summary(self, trainer_state=None) -> Path:
        summary: dict[str, object] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "max_process_rss_gb": round(self.max_process_rss_gb, 4),
            "max_system_used_gb": round(self.max_system_used_gb, 4),
        }
        if trainer_state is not None:
            summary["global_step"] = getattr(trainer_state, "global_step", None)
            summary["epoch"] = getattr(trainer_state, "epoch", None)
            summary["best_metric"] = getattr(trainer_state, "best_metric", None)
            summary["best_checkpoint"] = getattr(trainer_state, "best_model_checkpoint", None)
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return self.summary_path


def resolve_model_source() -> str:
    if MODEL_PATH_OVERRIDE:
        normalized = normalize_hf_model_path(MODEL_PATH_OVERRIDE)
        if normalized:
            return normalized
    cache_folder = "models--" + MODEL_NAME.replace("/", "--")
    snapshot = resolve_hf_snapshot(cache_folder)
    if snapshot:
        return snapshot
    return MODEL_NAME


def load_unsloth_model(fast_model_cls):
    common_kwargs = dict(
        model_name=resolve_model_source(),
        dtype=None,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
        token=HF_TOKEN or None,
    )
    try:
        return fast_model_cls.from_pretrained(**common_kwargs)
    except Exception:
        return fast_model_cls.from_pretrained(**common_kwargs, local_files_only=True)


def check_prerequisites() -> int:
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Training data not found: {TRAIN_DATA}")
        print("  Run training/prepare_stt_final_dataset.py first.")
        sys.exit(1)
    if not VAL_DATA.exists():
        print(f"[ERROR] Validation data not found: {VAL_DATA}")
        print("  Run training/prepare_stt_final_dataset.py first.")
        sys.exit(1)
    with open(TRAIN_DATA, encoding="utf-8") as handle:
        count = sum(1 for _ in handle)
    print(f"  Training examples: {count}")
    return count


def train() -> None:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import Trainer, TrainerCallback
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    class AudioSafeSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
            inputs["use_cache"] = False
            return Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    class MemorySnapshotCallback(TrainerCallback):
        def __init__(self, logger: MemoryMetricsLogger):
            self.logger = logger

        def on_train_begin(self, args, state, control, **kwargs):
            self.logger.snapshot("train_begin", step=state.global_step, epoch=state.epoch)
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.logger.snapshot("evaluate", step=state.global_step, epoch=state.epoch, extra={"eval_loss": metrics.get("eval_loss") if metrics else None})
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.logger.snapshot("train_end", step=state.global_step, epoch=state.epoch)
            self.logger.write_summary(state)
            return control

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"e4b_stt_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(exist_ok=True)
    memory_logger = MemoryMetricsLogger(run_dir, OUTPUT_DIR)

    model, processor = load_unsloth_model(FastVisionModel)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=FINETUNE_VISION_LAYERS,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
    )
    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    val_ds = load_dataset("json", data_files=str(VAL_DATA), split="train")
    data_collator = UnslothVisionDataCollator(model, processor)
    trainer = AudioSafeSFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=data_collator,
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
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
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
    stats = trainer.train()
    adapter_dir = OUTPUT_DIR / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    metrics_json, metrics_csv = save_training_metrics(trainer, run_dir, OUTPUT_DIR)
    memory_summary = memory_logger.write_summary(trainer.state)
    stats_path = OUTPUT_DIR / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": MODEL_NAME,
                "phase": "stt_final",
                "epochs": EPOCHS,
                "train_loss": stats.training_loss,
                "best_metric": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "run_dir": str(run_dir),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Adapter        : {adapter_dir}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best eval_loss : {trainer.state.best_metric}")
    print(f"Metrics JSON   : {metrics_json}")
    print(f"Metrics CSV    : {metrics_csv}")
    print(f"Memory summary : {memory_summary}")


if __name__ == "__main__":
    print("=" * 60)
    print("Gemma4GR - Train Final STT-only LoRA (Local)")
    print("=" * 60 + "\n")
    count = check_prerequisites()
    if count == 0:
        sys.exit(1)
    train()
