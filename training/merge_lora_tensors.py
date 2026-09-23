"""Merge plain LoRA adapters into base safetensors on the CPU, tensor by tensor.

    W_merged = W_base + sum_i scale_i * (B_i @ A_i)     (math in fp32, stored in the base dtype)

No model class, no Unsloth, no accelerate. The Unsloth/PEFT path loads the whole
fp16 model onto one GPU; E4B does not fit on a 16 GB card, so accelerate
offloads parameters to meta/CPU and the merge silently works on placeholders.
This path has no such failure mode: every weight is read from disk, and every
adapter module must match exactly one base weight of the right shape.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."
# Non-weight files copied from the base checkpoint so llama.cpp sees an ordinary HF dir.
SKIP_COPY = {".gitattributes", "README.md"}


def lora_scale(cfg: dict) -> float:
    r = cfg["r"]
    return cfg["lora_alpha"] / (r ** 0.5 if cfg.get("use_rslora") else r)


def check_plain_lora(cfg: dict, adapter_dir: Path) -> None:
    problems = []
    if cfg.get("use_dora"):
        problems.append("use_dora")
    if cfg.get("modules_to_save"):
        problems.append("modules_to_save")
    if cfg.get("bias", "none") != "none":
        problems.append(f"bias={cfg.get('bias')}")
    if cfg.get("fan_in_fan_out"):
        problems.append("fan_in_fan_out")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        problems.append("rank/alpha pattern")
    if problems:
        raise ValueError(f"{adapter_dir}: not a plain LoRA adapter ({', '.join(problems)})")


def load_adapter(adapter_dir: Path) -> tuple[float, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Return (scale, {base weight key: (A, B)}) for one adapter."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    check_plain_lora(cfg, adapter_dir)
    modules: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        keys = set(f.keys())
        for ka in sorted(k for k in keys if k.endswith(".lora_A.weight")):
            kb = ka.replace(".lora_A.weight", ".lora_B.weight")
            if kb not in keys or not ka.startswith(PREFIX):
                raise ValueError(f"{adapter_dir}: unexpected LoRA key {ka}")
            wkey = ka[len(PREFIX):].replace(".lora_A.weight", ".weight")
            modules[wkey] = (f.get_tensor(ka).float(), f.get_tensor(kb).float())
        other = {k for k in keys if ".lora_A." not in k and ".lora_B." not in k}
        if other:
            raise ValueError(f"{adapter_dir}: non-LoRA tensors present: {sorted(other)[:5]}")
    if not modules:
        raise ValueError(f"{adapter_dir}: no LoRA modules")
    return lora_scale(cfg), modules


def base_shards(base_dir: Path) -> list[Path]:
    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {base_dir}")
    return shards


def merge_to_dir(base_dir: Path, adapter_dirs: list[Path], out_dir: Path) -> dict:
    """Write merged safetensors (one output file per base shard) plus the base's
    non-weight files into out_dir. Returns a report."""
    adapters = [(d, *load_adapter(d)) for d in adapter_dirs]
    pending = {d: set(mods) for d, _, mods in adapters}
    changed = {str(d): 0 for d in adapter_dirs}
    merged_keys = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in base_shards(base_dir):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), "pt") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                w = f.get_tensor(key)
                hits = [(d, s, mods[key]) for d, s, mods in adapters if key in mods]
                if hits:
                    acc = w.float()
                    for d, scale, (a, b) in hits:
                        delta = (b @ a) * scale
                        if delta.shape != acc.shape:
                            raise ValueError(f"{d}: {key} delta {tuple(delta.shape)} != base {tuple(acc.shape)}")
                        if bool(delta.abs().max() > 0):
                            changed[str(d)] += 1
                        acc += delta
                        pending[d].discard(key)
                    w = acc.to(w.dtype)
                    merged_keys += 1
                tensors[key] = w.contiguous()
        save_file(tensors, str(out_dir / shard.name), metadata={**metadata, "format": "pt"})
        del tensors
    unmatched = {str(d): sorted(k)[:5] for d, k in pending.items() if k}
    if unmatched:
        raise ValueError(f"adapter modules with no base weight: {unmatched}")
    dead = [d for d, n in changed.items() if n == 0]
    if dead:
        raise ValueError(f"adapters that change no weight (all-zero deltas): {dead}")
    index = base_dir / "model.safetensors.index.json"
    for src in base_dir.iterdir():
        if src.is_file() and src.suffix != ".safetensors" and src.name not in SKIP_COPY:
            shutil.copy2(src, out_dir / src.name)
    if len(base_shards(base_dir)) > 1 and not index.is_file():
        raise FileNotFoundError(f"sharded base without index: {base_dir}")
    return {"merged_weight_tensors": merged_keys, "nonzero_modules_per_adapter": changed}
