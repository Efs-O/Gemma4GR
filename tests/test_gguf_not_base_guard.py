import json
import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import save_file

from training.gguf_not_base_guard import verify_gguf, verify_mmproj


OPS = {
    "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
    "o_proj": "attn_output", "gate_proj": "ffn_gate",
    "up_proj": "ffn_up", "down_proj": "ffn_down",
}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base.safetensors"
        self.adapter1 = self.root / "adapter1"
        self.adapter2 = self.root / "adapter2"
        self.adapter1.mkdir()
        self.adapter2.mkdir()
        self.base_weights = {}
        self.delta1 = {}
        self.delta2 = {}
        for layer in (0, 15, 30):
            for op in OPS:
                prefix = f"base_model.model.model.language_model.layers.{layer}.{'self_attn' if op in OPS and op.endswith('_proj') and op not in ('gate_proj','up_proj','down_proj') else 'mlp'}.{op}"
                hkey = prefix.removeprefix("base_model.model.") + ".weight"
                self.base_weights[hkey] = np.eye(32, dtype=np.float32)
                a = np.zeros((2, 32), dtype=np.float32); a[0, 0] = 1; a[1, 1] = 1
                b1 = np.zeros((32, 2), dtype=np.float32); b1[0, 0] = 0.8; b1[1, 1] = 0.8
                b2 = np.zeros((32, 2), dtype=np.float32); b2[0, 0] = -0.5; b2[1, 1] = -0.5
                keya, keyb = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
                self.delta1[prefix] = (b1 @ a) * 2.0
                self.delta2[prefix] = (b2 @ a) * 2.0
                if "ad1" not in self.__dict__: self.__dict__["ad1"] = {}
                if "ad2" not in self.__dict__: self.__dict__["ad2"] = {}
                self.ad1[keya] = a; self.ad1[keyb] = b1
                self.ad2[keya] = a; self.ad2[keyb] = b2
                self._mapping = getattr(self, "_mapping", {})
                self._mapping[prefix] = (hkey, f"blk.{layer}.{OPS[op]}.weight")
        save_file(self.base_weights, str(self.base))
        for folder, tensors in ((self.adapter1, self.ad1), (self.adapter2, self.ad2)):
            save_file(tensors, str(folder / "adapter_model.safetensors"))
            (folder / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_rslora": False}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_gguf(self, path, tuned=0, omit=None, quant=None, name="Synthetic"):
        writer = gguf.GGUFWriter(path, "gemma")
        writer.add_name(name)
        for prefix, (hkey, gname) in self._mapping.items():
            if gname == omit: continue
            tensor = self.base_weights[hkey].copy()
            if tuned >= 1: tensor += self.delta1[prefix]
            if tuned >= 2: tensor += self.delta2[prefix]
            tensor = tensor.astype(np.float32)
            qtype = quant if quant is not None else gguf.GGMLQuantizationType.F32
            if qtype == gguf.GGMLQuantizationType.F16:
                tensor = tensor.astype(np.float16)
            elif qtype != gguf.GGMLQuantizationType.F32:
                tensor = gguf.quants.quantize(tensor, qtype)
            raw_shape = (32, 34) if qtype == gguf.GGMLQuantizationType.Q8_0 else (32, 32)
            writer.add_tensor(gname, tensor, raw_shape=raw_shape, raw_dtype=qtype)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file()
        return Path(path)

    def test_tuned_gguf_passes(self):
        path = self._write_gguf(self.root / "tuned.gguf", tuned=1)
        report = verify_gguf(path, self.base, [self.adapter1])
        self.assertEqual(report["verdict"], "PASS")

    def test_tuned_f16_gguf_passes(self):
        path = self._write_gguf(self.root / "tuned-f16.gguf", tuned=1, quant=gguf.GGMLQuantizationType.F16)
        self.assertEqual(verify_gguf(path, self.base, [self.adapter1])["verdict"], "PASS")

    def test_base_gguf_fails_even_with_tuned_name(self):
        path = self._write_gguf(self.root / "base.gguf", name="Gemma4GR Tuned")
        self.assertEqual(verify_gguf(path, self.base, [self.adapter1])["verdict"], "FAIL")

    def test_base_gguf_fails_with_base_name(self):
        path = self._write_gguf(self.root / "base-name.gguf", name="Synthetic Base")
        self.assertEqual(verify_gguf(path, self.base, [self.adapter1])["verdict"], "FAIL")

    def test_missing_probe_fails(self):
        path = self._write_gguf(self.root / "missing.gguf", tuned=1, omit="blk.0.attn_q.weight")
        with self.assertRaises(Exception): verify_gguf(path, self.base, [self.adapter1])

    def test_too_few_probes_fails(self):
        one = self.root / "small"
        one.mkdir()
        key = next(iter(self.ad1))
        save_file({key: self.ad1[key]}, str(one / "adapter_model.safetensors"))
        with self.assertRaises(Exception): verify_gguf(self._write_gguf(self.root / "few.gguf", tuned=1), self.base, [one])

    def test_two_adapter_additive_delta(self):
        path = self._write_gguf(self.root / "sum.gguf", tuned=2)
        self.assertEqual(verify_gguf(path, self.base, [self.adapter1, self.adapter2])["verdict"], "PASS")
        self.assertEqual(verify_gguf(path, self.base, [self.adapter1])["verdict"], "FAIL")

    def test_same_quant_base_bytes_fail(self):
        qtype = gguf.GGMLQuantizationType.Q8_0
        base = self._write_gguf(self.root / "qbase.gguf", quant=qtype)
        tuned = self._write_gguf(self.root / "qtuned.gguf", tuned=1, quant=qtype)
        report = verify_gguf(tuned, self.base, [self.adapter1], base_gguf=base)
        self.assertEqual(report["verdict"], "PASS")
        identical = self._write_gguf(self.root / "qidentical.gguf", quant=qtype)
        self.assertEqual(verify_gguf(identical, self.base, [self.adapter1], base_gguf=base)["verdict"], "FAIL")

    def test_mmproj_stock_pass_altered_fails(self):
        stock = gguf.GGUFWriter(str(self.root / "stock.gguf"), "gemma")
        stock.add_name("Projector"); stock.add_tensor("proj.weight", np.eye(32,dtype=np.float32))
        stock.write_header_to_file(); stock.write_kv_data_to_file(); stock.write_tensors_to_file()
        altered = gguf.GGUFWriter(str(self.root / "altered.gguf"), "gemma")
        altered.add_name("Projector"); changed=np.eye(32,dtype=np.float32); changed[0,0]=2
        altered.add_tensor("proj.weight", changed)
        altered.write_header_to_file(); altered.write_kv_data_to_file(); altered.write_tensors_to_file()
        self.assertEqual(verify_mmproj(self.root / "stock.gguf", self.root / "stock.gguf", [self.adapter1])["verdict"], "PASS")
        self.assertEqual(verify_mmproj(self.root / "altered.gguf", self.root / "stock.gguf", [self.adapter1])["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
