"""Guarded launcher for the single Gemma4GR v3 QA run."""
from __future__ import annotations

import argparse
import json
import math
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


def drop_string_columns(dataset):
    first = dataset[0]
    text_cols = [name for name in dataset.column_names if isinstance(first[name], str)]
    return dataset.remove_columns(text_cols) if text_cols else dataset


def disable_use_cache(model) -> None:
    modules = model.modules() if hasattr(model, "modules") else [model]
    for module in [model, *modules]:
        config = getattr(module, "config", None)
        if config is not None:
            config.use_cache = False
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                text_config.use_cache = False
        for attr in ("base_model", "model", "language_model"):
            child = getattr(module, attr, None)
            if child is not None and child is not module and not hasattr(model, "modules"):
                disable_use_cache(child)


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
    # Unsloth must be imported before trl, otherwise SFTTrainer is TRL's unpatched class
    # (its tokenize path appended <unk> after <turn|> in the G6 preflight).
    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    if SFTTrainer.__name__ != "UnslothSFTTrainer":
        raise RuntimeError(f"SFTTrainer is not Unsloth-patched ({SFTTrainer.__module__})")

    # sdpa, not Unsloth's default flex_attention: flex compiles on first call, and that forward's saved tensors
    # differ from the checkpoint recompute (CheckpointError, G6 preflight 4-7). sdpa: identical loss/grads to the
    # KV-cache reference path (G6_grad_probe_sdpa.log). Gemma 4's only softcap is on final logits, not attention.
    model, processor = FastVisionModel.from_pretrained(model_name=cfg["base_model"], max_seq_length=cfg["max_length"],
        load_in_4bit=True, dtype=None, trust_remote_code=True, attn_implementation="sdpa")
    if model.config._attn_implementation != "sdpa":
        raise RuntimeError(f"attention implementation is {model.config._attn_implementation}, expected sdpa")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model = FastVisionModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing="unsloth")
    # Gemma 4 E4B shares K/V across its last 18 layers. With use_cache=True a real DynamicCache is built and
    # checkpoint recompute appends to it twice (CheckpointError, G6 preflight 4/5). With use_cache=False,
    # Unsloth's KV-sharing carrier is used instead; G6 grad probe: loss and LoRA grads identical to the cache path.
    disable_use_cache(model)
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
            # SFTTrainer init clears the decoder layers' checkpointing flags; with this set, train() re-enables them
            # (non-reentrant, as Unsloth's Gemma 4 shared-KV patch requires). Off: 21 GB peak on the longest row;
            # on: 12.9 GB with identical losses (G6 mem_probe4 / preflight 10).
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=1, report_to="none", remove_unused_columns=False))
    # G6 run 1 slowed from ~4.5 to ~60 s/step after the first eval: the eval forward grew the cached pool to the whole
    # card and Windows spilled VRAM to system RAM. Release the cache after every eval/save.
    from transformers import TrainerCallback

    class ReleaseCudaCache(TrainerCallback):
        def _release(self, *args, **kwargs):
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        on_evaluate = on_save = _release

    trainer.add_callback(ReleaseCudaCache())
    print("pre-mask columns:", trainer.train_dataset.column_names, "ids tail:", trainer.train_dataset[0]["input_ids"][-5:],
          "collator:", type(trainer.data_collator).__name__, flush=True)
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
    print("post-mask ids tail:", trainer.train_dataset[0]["input_ids"][-5:], "labels tail:", trainer.train_dataset[0]["labels"][-5:],
          "collator:", type(trainer.data_collator).__name__, flush=True)
    # remove_unused_columns=False keeps the raw "text" column, which the collator cannot tensorize.
    trainer.train_dataset = drop_string_columns(trainer.train_dataset)
    trainer.eval_dataset = drop_string_columns(trainer.eval_dataset)
    print("train columns:", trainer.train_dataset.column_names, flush=True)
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
        seen_cache_args = []
        text_models = [m for m in trainer.model.modules() if type(m).__name__ == "Gemma4TextModel"]
        def record_cache_args(module, args_, kwargs):
            seen_cache_args.append((kwargs.get("use_cache"), module.config.use_cache, type(kwargs.get("past_key_values")).__name__))
        hooks = [m.register_forward_pre_hook(record_cache_args, with_kwargs=True) for m in text_models]
        try:
            trainer.train()
        finally:
            for hook in hooks:
                hook.remove()
            print("text model cache args (kwarg, config, past_key_values):", sorted(set(map(str, seen_cache_args))), flush=True)
        losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
        print("step losses:", losses, flush=True)
        if not losses or not all(math.isfinite(x) for x in losses):
            raise AssertionError(f"Preflight losses not finite: {losses}")
        train_peak = torch.cuda.max_memory_allocated()
        print("5-step peak_cuda_memory_gb:", round(train_peak / 2**30, 2), flush=True)
        # Worst case for memory, through trainer.train() (checkpointing is only active inside train()):
        # 2 optimizer steps on the 8 longest training rows.
        lengths = [len(ids) for ids in trainer.train_dataset["input_ids"]]
        longest = sorted(range(len(lengths)), key=lambda i: lengths[i])[-8:]
        trainer.train_dataset = trainer.train_dataset.select(longest)
        trainer.args.max_steps = 2
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        print("longest rows tokens:", sorted(lengths[i] for i in longest), "peak_cuda_memory_gb:", round(peak / 2**30, 2),
              "of", round(total / 2**30, 2), flush=True)
        (run_output / "memory.json").write_text(json.dumps({"five_step_peak_bytes": train_peak, "longest_rows_peak_bytes": peak,
            "longest_row_tokens": max(lengths), "device_total_bytes": total}, indent=2), encoding="utf-8")
        if peak > 0.9 * total:
            raise AssertionError(f"Peak memory {peak} exceeds 90% of device memory {total}")
        return
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(str(run_output / "run"))
        if checkpoint is None:
            raise ValueError("--resume requested but no checkpoint exists under output/v3")
    else:
        checkpoint = None
    # Cap the caching allocator below the card so a full pool frees its cache and retries (or fails loudly)
    # instead of spilling into system RAM.
    torch.cuda.set_per_process_memory_fraction(0.94, 0)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.save_pretrained(output / "lora_adapter")
    processor.save_pretrained(output / "lora_adapter")
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "memory.json").write_text(json.dumps({"peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
