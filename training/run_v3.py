"""Guarded launcher for the single Gemma4GR v3 QA run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "private" / "v3_train_config.json"
REQUIRED = ("base_model", "stt_adapter", "train_data", "val_data", "output_dir", "max_length",
            "epochs", "learning_rate", "seed", "data_seed", "save_steps", "batch_size",
            "gradient_accumulation_steps")
FORBIDDEN = ("persona", "_private_quarantine", "e4b efso")


def read_config(path: Path = CONFIG_PATH) -> dict:
    if not path.is_file():
        raise ValueError(f"Required config missing: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in cfg or cfg[key] in (None, "")]
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))
    for key in REQUIRED[:5]:
        raw = Path(cfg[key]).expanduser()
        resolved = (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
        low = str(resolved).lower()
        if any(term in low for term in FORBIDDEN):
            raise ValueError(f"Forbidden path for {key}")
        cfg[key] = str(resolved)
    for key in ("base_model", "stt_adapter", "train_data", "val_data"):
        if not Path(cfg[key]).exists():
            raise ValueError(f"Missing configured input: {key}={cfg[key]}")
    out = Path(cfg["output_dir"])
    if out == ROOT / "output" / "v2" or ROOT / "output" / "v2" in out.parents:
        raise ValueError("v3 output cannot target output/v2")
    if not (ROOT / "output" / "v3") in out.parents and out != ROOT / "output" / "v3":
        raise ValueError("output_dir must be under output/v3")
    cfg["output_dir"] = str(out)
    return cfg


def strip_one_bos(text: str) -> str:
    return re.sub(r"^<bos>", "", text, count=1)


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
    cfg = read_config()
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output dir is non-empty; pass --resume only for an existing run")
    print(json.dumps(cfg, ensure_ascii=False, indent=2), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    model, processor = FastVisionModel.from_pretrained(model_name=cfg["base_model"], max_seq_length=cfg["max_length"],
        load_in_4bit=True, dtype=None, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, cfg["stt_adapter"], is_trainable=False)
    model = FastVisionModel.get_peft_model(model, r=32, lora_alpha=64, lora_dropout=0.05,
        bias="none", finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing="unsloth")
    tokenizer = get_chat_template(processor.tokenizer, chat_template="gemma-4")
    # G4 rows carry a literal BOS. Keep it and disable automatic BOS insertion.
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    processor.tokenizer = tokenizer
    ds = load_dataset("json", data_files={"train": cfg["train_data"], "validation": cfg["val_data"]})
    def normalize(row):
        row["text"] = row["text"]
        ids = tokenizer(row["text"], add_special_tokens=True, truncation=False)["input_ids"]
        if tokenizer.bos_token_id is not None and (not ids or ids[0] != tokenizer.bos_token_id or ids.count(tokenizer.bos_token_id) != 1):
            raise ValueError("training row must tokenize to exactly one leading BOS")
        if len(ids) > cfg["max_length"]:
            raise ValueError(f"row exceeds max_length ({len(ids)} > {cfg['max_length']})")
        return row
    train = ds["train"].map(normalize)
    val = ds["validation"].map(normalize)
    steps = 5 if args.preflight else -1
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train, eval_dataset=val,
        args=SFTConfig(output_dir=str(output / "run"), dataset_text_field="text", max_length=cfg["max_length"],
            num_train_epochs=cfg["epochs"], max_steps=steps, learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            seed=cfg["seed"], data_seed=cfg["data_seed"], eval_strategy="steps", eval_steps=cfg["save_steps"],
            save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=None,
            load_best_model_at_end=False, optim="adamw_8bit", lr_scheduler_type="linear", warmup_steps=5,
            weight_decay=0.001, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True, report_to="none", remove_unused_columns=False))
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
    if args.preflight:
        batch = next(iter(trainer.get_train_dataloader()))
        print("preflight batch label tokens:", int((batch["labels"] != -100).sum()), flush=True)
        trainer.args.max_steps = 5
        trainer.train()
        return
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(str(output / "run"))
        if checkpoint is None:
            raise ValueError("--resume requested but no checkpoint exists under output/v3")
    else:
        checkpoint = None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.save_pretrained(output / "lora_adapter")
    processor.save_pretrained(output / "lora_adapter")


if __name__ == "__main__":
    main()
