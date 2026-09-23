"""Tensor-level verification that LoRA deltas reached GGUF output weights."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open
import gguf

from training.gguf_export_utils import ExportValidationError


_PROJ = {
    "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
    "o_proj": "attn_output", "gate_proj": "ffn_gate",
    "up_proj": "ffn_up", "down_proj": "ffn_down",
}
_LORA_KEY = re.compile(r"^(.*)\.lora_([AB])\.weight$")


def _hf_key(prefix: str) -> str:
    return prefix.removeprefix("base_model.model.") + ".weight"


def _gguf_name(prefix: str) -> str:
    m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([^.]+)$", prefix)
    if not m or m.group(2) not in _PROJ:
        raise ExportValidationError(f"unmapped adapter tensor: {prefix}")
    return f"blk.{m.group(1)}.{_PROJ[m.group(2)]}.weight"


def _weights(path: Path):
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ExportValidationError(f"no safetensors found: {path}")
    return files


def _adapter_map(adapter: Path):
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    rank, alpha = int(config["r"]), float(config["lora_alpha"])
    scale = alpha / (rank ** 0.5 if config.get("use_rslora") else rank)
    paths = sorted(adapter.glob("adapter_model*.safetensors"))
    if not paths:
        raise ExportValidationError(f"no adapter safetensors in {adapter}")
    result = {}
    for path in paths:
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                match = _LORA_KEY.match(key)
                if not match:
                    continue
                prefix, side = match.groups()
                item = result.setdefault(prefix, {"scale": scale})
                item[side] = f.get_tensor(key).astype(np.float32, copy=False)
    if not result:
        raise ExportValidationError(f"no LoRA A/B keys in {adapter}")
    return result


def _reader(path: Path):
    reader = gguf.GGUFReader(str(path))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    return reader, tensors


def _raw(tensor):
    return np.asarray(tensor.data).tobytes()


def _get_base_tensor(base_paths, key):
    for path in base_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key).float().numpy()
    raise ExportValidationError(f"base safetensor is missing {key}")


def build_probe_deltas(base_safetensors: Path, adapters: list[Path], min_probes=12):
    adapter_maps = [_adapter_map(adapter) for adapter in adapters]
    prefixes = sorted(set().union(*(set(m) for m in adapter_maps)))
    if len(prefixes) < min_probes:
        raise ExportValidationError(f"fewer than {min_probes} adapter-targeted probes available")
    base_paths = _weights(base_safetensors)
    candidates = []
    for prefix in prefixes:
        layer = re.search(r"layers\.(\d+)", prefix)
        ggname = _gguf_name(prefix)
        base_key = _hf_key(prefix)
        base = _get_base_tensor(base_paths, base_key)
        delta = None
        for amap in adapter_maps:
            pair = amap.get(prefix)
            if pair is None:
                continue
            if "A" not in pair or "B" not in pair:
                raise ExportValidationError(f"incomplete LoRA pair: {prefix}")
            part = (pair["B"] @ pair["A"]) * pair["scale"]
            delta = part if delta is None else delta + part
        if delta is None:
            raise ExportValidationError(f"no adapter delta for {prefix}")
        if base.shape != delta.shape:
            raise ExportValidationError(f"base/delta shape mismatch for {prefix}: {base.shape} vs {delta.shape}")
        rel = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(base.ravel()), 1e-30))
        candidates.append({"prefix": prefix, "name": ggname, "base_key": base_key,
                           "relative_delta_norm": rel,
                           "layer": int(layer.group(1)) if layer else -1,
                           "projection": ggname.split(".")[2]})
        del base, delta
    # First guarantee depth and operation diversity; fill remaining slots by the
    # largest relative delta norm, keeping probes spread over early/middle/late.
    selected = []
    for depth in ("early", "middle", "late"):
        band = {"early": lambda x: x["layer"] < 14,
                "middle": lambda x: 14 <= x["layer"] < 28,
                "late": lambda x: x["layer"] >= 28}[depth]
        for op in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            options = [x for x in candidates if band(x) and x["projection"] == op]
            if options:
                selected.append(max(options, key=lambda x: x["relative_delta_norm"]))
    selected_names = {x["prefix"] for x in selected}
    selected.extend(sorted((x for x in candidates if x["prefix"] not in selected_names),
                           key=lambda x: x["relative_delta_norm"], reverse=True)[:max(0, 12-len(selected))])
    if len(selected) < 12:
        raise ExportValidationError(f"only {len(selected)} mappable probes available")
    for probe in selected:
        delta = None
        for amap in adapter_maps:
            pair = amap.get(probe["prefix"])
            if pair is not None:
                part = (pair["B"] @ pair["A"]) * pair["scale"]
                delta = part if delta is None else delta + part
        if delta is None:
            raise ExportValidationError(f"cannot recompute selected delta: {probe['prefix']}")
        probe["delta"] = delta
    return selected, base_paths


def verify_gguf(gguf_path: Path, base_safetensors: Path, adapters: list[Path],
                base_gguf: Path | None = None, min_probes: int = 12):
    probes, base_paths = build_probe_deltas(base_safetensors, adapters, min_probes)
    _, tested = _reader(gguf_path)
    same = _reader(base_gguf)[1] if base_gguf else {}
    results = []
    for probe in probes:
        name = probe["name"]
        if name not in tested:
            raise ExportValidationError(f"probe tensor missing from GGUF: {name}")
        tensor = tested[name]
        deq = gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)
        base = _get_base_tensor(base_paths, probe["base_key"])
        if deq.shape != base.shape:
            if deq.T.shape == base.shape:
                deq = deq.T
            else:
                raise ExportValidationError(f"GGUF/base shape mismatch for {name}: {deq.shape} vs {base.shape}")
        expected = base + probe["delta"]
        if expected.shape != deq.shape:
            raise ExportValidationError(f"merged shape mismatch for {name}")
        d_base = float(np.linalg.norm((deq - base).ravel()))
        d_merged = float(np.linalg.norm((deq - expected).ravel()))
        identical = False
        if base_gguf:
            if name not in same:
                raise ExportValidationError(f"probe tensor missing from comparison base GGUF: {name}")
            identical = _raw(tensor) == _raw(same[name])
        passed = d_merged < d_base and not identical
        results.append({"name": name, "quant_type": str(tensor.tensor_type),
                        "relative_delta_norm": probe["relative_delta_norm"],
                        "d_base": d_base, "d_merged": d_merged,
                        "ratio": d_merged / max(d_base, 1e-30),
                        "byte_identical": identical, "pass": passed})
    if len(results) < min_probes:
        raise ExportValidationError(f"only {len(results)} probes evaluated; need {min_probes}")
    return {"gguf": str(gguf_path), "verdict": "PASS" if all(x["pass"] for x in results) else "FAIL",
            "probe_count": len(results), "probes": results}


def verify_mmproj(mmproj: Path, stock_mmproj: Path, adapters: list[Path]):
    """Text-only adapters require projector tensor bytes to match stock exactly."""
    if any(_adapter_targets_vision_or_audio(p) for p in adapters):
        raise ExportValidationError("audio/vision LoRA deltas require projector probes; unsupported mapping")
    _, actual = _reader(mmproj)
    _, expected = _reader(stock_mmproj)
    if set(actual) != set(expected):
        raise ExportValidationError("mmproj tensor names differ from the stock projector")
    mismatched = [name for name in sorted(actual) if _raw(actual[name]) != _raw(expected[name])]
    return {"mmproj": str(mmproj), "verdict": "FAIL" if mismatched else "PASS",
            "tensor_count": len(actual), "mismatched": mismatched}


def _adapter_targets_vision_or_audio(adapter: Path) -> bool:
    for part in _adapter_map(adapter):
        lower = part.lower()
        if any(token in lower for token in ("vision", "projector", "audio", "speech")):
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--base-safetensors", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path, nargs="+")
    parser.add_argument("--base-gguf", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-probes", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        report = verify_gguf(args.gguf, args.base_safetensors, args.adapter,
                             args.base_gguf, args.min_probes)
    except Exception as exc:
        report = {"gguf": str(args.gguf), "verdict": "FAIL", "error": str(exc), "probes": []}
        if args.report:
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
