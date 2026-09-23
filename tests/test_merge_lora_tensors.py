import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load, save_file

from training.merge_lora_tensors import merge_to_dir

Q = "model.language_model.layers.0.self_attn.q_proj"
V = "model.language_model.layers.0.self_attn.v_proj"


def write_base(d: Path) -> dict:
    torch.manual_seed(0)
    base = {f"{Q}.weight": torch.randn(8, 6).to(torch.bfloat16),
            f"{V}.weight": torch.randn(4, 6).to(torch.bfloat16),
            "model.language_model.norm.weight": torch.ones(6, dtype=torch.bfloat16)}
    d.mkdir()
    save_file(base, str(d / "model.safetensors"))
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "README.md").write_text("x", encoding="utf-8")
    return base


def write_adapter(d: Path, modules: dict, r=2, alpha=4, **cfg) -> dict:
    """modules: {module name: out_features}; returns {weight key: delta}."""
    d.mkdir()
    tensors, deltas = {}, {}
    for mod, out in modules.items():
        a, b = torch.randn(r, 6), torch.randn(out, r)
        tensors[f"base_model.model.{mod}.lora_A.weight"] = a
        tensors[f"base_model.model.{mod}.lora_B.weight"] = b
        deltas[f"{mod}.weight"] = (b @ a) * (alpha / r)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "bias": "none", **cfg}), encoding="utf-8")
    return deltas


def read(path: Path) -> dict:
    return load(path.read_bytes())  # no mmap: Windows cannot delete a mapped file


class TensorMergeTests(unittest.TestCase):
    def test_merged_weights_equal_base_plus_all_deltas(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base = write_base(t / "base")
            d1 = write_adapter(t / "a1", {Q: 8, V: 4})
            d2 = write_adapter(t / "a2", {Q: 8})
            report = merge_to_dir(t / "base", [t / "a1", t / "a2"], t / "out")
            merged = read(t / "out" / "model.safetensors")
            want_q = (base[f"{Q}.weight"].float() + d1[f"{Q}.weight"] + d2[f"{Q}.weight"]).to(torch.bfloat16)
            want_v = (base[f"{V}.weight"].float() + d1[f"{V}.weight"]).to(torch.bfloat16)
            self.assertTrue(torch.equal(merged[f"{Q}.weight"], want_q))
            self.assertTrue(torch.equal(merged[f"{V}.weight"], want_v))
            self.assertTrue(torch.equal(merged["model.language_model.norm.weight"], base["model.language_model.norm.weight"]))
            self.assertEqual(merged[f"{Q}.weight"].dtype, torch.bfloat16)
            self.assertEqual(report["merged_weight_tensors"], 2)
            self.assertTrue((t / "out" / "config.json").is_file())
            self.assertFalse((t / "out" / "README.md").exists())

    def test_module_without_base_weight_fails(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_base(t / "base")
            write_adapter(t / "a", {"model.language_model.layers.9.mlp.up_proj": 8})
            with self.assertRaisesRegex(ValueError, "no base weight"):
                merge_to_dir(t / "base", [t / "a"], t / "out")

    def test_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_base(t / "base")
            write_adapter(t / "a", {V: 8})
            with self.assertRaisesRegex(ValueError, "delta"):
                merge_to_dir(t / "base", [t / "a"], t / "out")

    def test_all_zero_adapter_fails(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_base(t / "base")
            write_adapter(t / "a", {Q: 8})
            p = t / "a" / "adapter_model.safetensors"
            tensors = read(p)
            save_file({k: torch.zeros_like(v) if "lora_B" in k else v for k, v in tensors.items()}, str(p))
            with self.assertRaisesRegex(ValueError, "change no weight"):
                merge_to_dir(t / "base", [t / "a"], t / "out")

    def test_non_plain_lora_rejected(self):
        for cfg in ({"use_dora": True}, {"modules_to_save": ["lm_head"]}, {"bias": "all"}):
            with self.subTest(cfg=cfg), tempfile.TemporaryDirectory() as t:
                t = Path(t)
                write_base(t / "base")
                write_adapter(t / "a", {Q: 8}, **cfg)
                with self.assertRaisesRegex(ValueError, "not a plain LoRA"):
                    merge_to_dir(t / "base", [t / "a"], t / "out")

    def test_rslora_scale(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base = write_base(t / "base")
            d = write_adapter(t / "a", {Q: 8}, r=4, alpha=4, use_rslora=True)
            merge_to_dir(t / "base", [t / "a"], t / "out")
            merged = read(t / "out" / "model.safetensors")
            # write_adapter scales by alpha/r = 1; rslora scale is alpha/sqrt(r) = 2
            want = (base[f"{Q}.weight"].float() + 2 * d[f"{Q}.weight"]).to(torch.bfloat16)
            self.assertTrue(torch.equal(merged[f"{Q}.weight"], want))


if __name__ == "__main__":
    unittest.main()
