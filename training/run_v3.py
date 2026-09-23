"""Guarded launcher for the single Gemma4GR v3 QA run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "private" / "v3_train_config.json"
REQUIRED = ("base_model", "train_data", "val_data", "output_dir", "max_length",
            "epochs", "learning_rate", "seed", "data_seed", "save_steps", "batch_size",
            "gradient_accumulation_steps", "lora_r", "lora_alpha", "lora_dropout",
            "warmup_steps", "weight_decay", "scheduler", "optimizer", "logging_steps")
FORBIDDEN = ("persona", "_private_quarantine", "e4b efso")


def read_config(path: Path = CONFIG_PATH) -> dict:
    if not path.is_file():
        raise ValueError(f"Required config missing: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in cfg or cfg[key] in (None, "")]
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))
    for key in ("base_model", "train_data", "val_data", "output_dir"):
        raw = Path(cfg[key]).expanduser()
        resolved = (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
        low = str(resolved).lower()
        if any(term in low for term in FORBIDDEN):
            raise ValueError(f"Forbidden path for {key}")
        cfg[key] = str(resolved)
    for key in ("base_model", "train_data", "val_data"):
        if not Path(cfg[key]).exists():
            raise ValueError(f"Missing configured input: {key}={cfg[key]}")
    out = Path(cfg["output_dir"])
    if out == ROOT / "output" / "v2" or ROOT / "output" / "v2" in out.parents:
        raise ValueError("v3 output cannot target output/v2")
    if not (ROOT / "output" / "v3") in out.parents and out != ROOT / "output" / "v3":
        raise ValueError("output_dir must be under output/v3")
    cfg["output_dir"] = str(out)
    merge_adapter = cfg.get("stt_adapter_for_merge")
    if merge_adapter:
        merge_path = Path(merge_adapter).expanduser().resolve()
        if any(term in str(merge_path).lower() for term in FORBIDDEN):
            raise ValueError("Forbidden path for stt_adapter_for_merge")
        if not merge_path.exists():
            raise ValueError(f"Missing configured input: stt_adapter_for_merge={merge_path}")
        cfg["stt_adapter_for_merge"] = str(merge_path)
    return cfg


def strip_one_bos(text: str) -> str:
    return re.sub(r"^<bos>", "", text, count=1)


def prepare_row(text: str) -> str:
    if text.endswith("\n"):
        text = text[:-1]
    if not text.endswith("<turn|>"):
        raise ValueError("training row must end with <turn|> after removing one trailing newline")
    return text


def preflight_output_dir(output: Path, stamp: str | None = None) -> Path:
    instant = datetime.now()
    if stamp is not None:
        instant = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    candidate = output.parent / f"preflight_{instant.strftime('%Y%m%d_%H%M%S')}"
    while candidate.exists():
        instant += timedelta(seconds=1)
        candidate = output.parent / f"preflight_{instant.strftime('%Y%m%d_%H%M%S')}"
    return candidate


def validate_gpu_visibility(value: str | None) -> None:
    if value is None or not re.fullmatch(r"\d", value):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one GPU index (0-9) before --preflight, --train, or --resume")


def response_labels(input_ids, tokenizer):
    """Mask through the assistant header; supervise answer and final EOS."""
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    marker = "<|turn>model\n"
    at = text.find(marker)
    if at < 0:
        raise ValueError("Rendered example has no model turn")
    prefix_ids = tokenizer(text[:at + len(marker)], add_special_tokens=False)["input_ids"]
    labels = list(input_ids)
    labels[:len(prefix_ids)] = [-100] * len(prefix_ids)
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    validate_gpu_visibility(os.environ.get("CUDA_VISIBLE_DEVICES"))
    cfg = read_config()
    output = Path(cfg["output_dir"])
    run_output = preflight_output_dir(output) if args.preflight else output
    if not args.preflight and output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output dir is non-empty; pass --resume only for an existing run")
    print(json.dumps(cfg, ensure_ascii=False, indent=2), flush=True)
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    model, processor = FastVisionModel.from_pretrained(model_name=cfg["base_model"], max_seq_length=cfg["max_length"],
        load_in_4bit=True, dtype=None, trust_remote_code=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model = FastVisionModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing="unsloth")
    tokenizer = get_chat_template(processor.tokenizer, chat_template="gemma-4")
    # G4 rows carry a literal BOS. Keep it and disable automatic BOS insertion.
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    processor.tokenizer = tokenizer
    ds = load_dataset("json", data_files={"train": cfg["train_data"], "validation": cfg["val_data"]})
    def normalize(row):
        row["text"] = prepare_row(row["text"])
        ids = tokenizer(text=row["text"])["input_ids"]
        if tokenizer.bos_token_id is not None and (not ids or ids[0] != tokenizer.bos_token_id or ids.count(tokenizer.bos_token_id) != 1):
            raise ValueError("training row must tokenize to exactly one leading BOS")
        if len(ids) > cfg["max_length"]:
            raise ValueError(f"row exceeds max_length ({len(ids)} > {cfg['max_length']})")
        return row
    train = ds["train"].map(normalize)
    val = ds["validation"].map(normalize)
    train_texts = train["text"]
    steps = 5 if args.preflight else -1
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train, eval_dataset=val,
        args=SFTConfig(output_dir=str(run_output / "run"), dataset_text_field="text", max_length=cfg["max_length"],
            num_train_epochs=cfg["epochs"], max_steps=steps, learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            seed=cfg["seed"], data_seed=cfg["data_seed"], eval_strategy="steps", eval_steps=cfg["save_steps"],
            save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=None,
            load_best_model_at_end=cfg.get("load_best_model_at_end", False), optim=cfg["optimizer"], lr_scheduler_type=cfg["scheduler"], warmup_steps=cfg["warmup_steps"],
            weight_decay=cfg["weight_decay"], logging_steps=cfg["logging_steps"], eos_token="<turn|>",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True, report_to="none", remove_unused_columns=False))
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
    trainable = [(name, p.numel()) for name, p in trainer.model.named_parameters() if p.requires_grad]
    trainable_count = sum(count for _, count in trainable)
    forbidden_trainable = [name for name, _ in trainable if any(x in name.lower() for x in ("vision", "audio", "embed_"))]
    if trainable_count <= 0 or forbidden_trainable:
        raise AssertionError(f"Invalid trainable parameters: count={trainable_count}, forbidden={forbidden_trainable}")
    (run_output / "trainable_parameters.json").write_text(json.dumps({"count": trainable_count, "names": [n for n, _ in trainable]}, indent=2), encoding="utf-8")
    if args.preflight:
        dump = []
        for index in range(min(3, len(trainer.train_dataset))):
            feature = trainer.train_dataset[index]
            batch = trainer.data_collator([feature])
            ids = batch["input_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            supervised_ids = [token for token in labels if token != -100]
            raw = train_texts[index]
            expected = raw.rsplit("<|turn>model\n", 1)[-1]
            supervised = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            record = {"n_tokens": len(ids), "n_bos": ids.count(2),
                      "first_5_ids": {"ids": ids[:5], "decoded": tokenizer.decode(ids[:5], skip_special_tokens=False)},
                      "last_5_ids": {"ids": ids[-5:], "decoded": tokenizer.decode(ids[-5:], skip_special_tokens=False)},
                      "supervised_text": supervised, "expected_supervised": expected,
                      "supervised_equals_expected": supervised == expected,
                      "n_masked": sum(token == -100 for token in labels)}
            if record["n_bos"] != 1 or ids[-1] != 106 or not record["supervised_equals_expected"]:
                raise AssertionError(f"Preflight labels failed for row {index}: {record}")
            dump.append(record)
        (run_output / "labels_dump.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print("trainable parameters:", trainable_count, flush=True)
        trainer.args.max_steps = 5
        trainer.args.logging_steps = 1
        trainer.train()
        for row in trainer.state.log_history:
            if "loss" in row:
                print("step loss:", row.get("step"), row["loss"], flush=True)
        return
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(str(run_output / "run"))
        if checkpoint is None:
            raise ValueError("--resume requested but no checkpoint exists under output/v3")
    else:
        checkpoint = None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.save_pretrained(output / "lora_adapter")
    processor.save_pretrained(output / "lora_adapter")
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "memory.json").write_text(json.dumps({"peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
