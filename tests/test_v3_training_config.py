import json
import tempfile
import unittest
from pathlib import Path

from training.run_v3 import (ROOT, preflight_output_dir, prepare_row, read_config,
                             response_labels, strip_one_bos, validate_gpu_visibility)


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class TrainingConfigTests(unittest.TestCase):
    def test_missing_required_config_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required config keys"):
                read_config(p)

    def test_single_bos(self):
        self.assertEqual(strip_one_bos("<bos><bos>x"), "<bos>x")
        self.assertEqual(strip_one_bos("x"), "x")

    def test_prepare_row_removes_one_trailing_newline(self):
        self.assertEqual(prepare_row("question<turn|>\n"), "question<turn|>")
        with self.assertRaisesRegex(ValueError, "must end with <turn\|>"):
            prepare_row("question<turn|>\n\n")

    def test_gpu_visibility_guard(self):
        for value in (None, "", "00", "0,1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                validate_gpu_visibility(value)
        validate_gpu_visibility("0")

    def test_preflight_uses_separate_timestamped_dir(self):
        output = ROOT / "output" / "v3" / "qa"
        self.assertEqual(preflight_output_dir(output, "20260923_235959"),
                         output.parent / "preflight_20260923_235959")
        self.assertNotEqual(preflight_output_dir(output, "20260923_235959"), output)

    def test_recipe_keys_required(self):
        required = ("lora_r", "lora_alpha", "lora_dropout", "warmup_steps", "weight_decay",
                    "scheduler", "optimizer", "logging_steps")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"base_model": "", **{key: 0 for key in required}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required config keys"):
                read_config(p)

    def test_response_masking(self):
        text = "<bos><|turn>user\nquestion<turn|>\n<|turn>model\nanswer<turn|>"
        labels = response_labels([ord(c) for c in text], FakeTokenizer())
        supervised = "".join(chr(x) for x in labels if x != -100)
        self.assertEqual(supervised, "answer<turn|>")

    def test_config_path_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = []
            for name in ("base", "adapter", "train", "val"):
                p = root / name
                p.mkdir()
                (p / "config.json").write_text("{}", encoding="utf-8")
                inputs.append(str(p))
            cfg = dict(zip(("base_model", "train_data", "val_data"), (inputs[0], inputs[2], inputs[3])))
            cfg["stt_adapter_for_merge"] = inputs[1]
            cfg.update(output_dir=str(ROOT / "output" / "v3" / "unit-test"), max_length=2048, epochs=1,
                       learning_rate=2e-5, seed=3407, data_seed=3407, save_steps=200,
                       batch_size=1, gradient_accumulation_steps=4, lora_r=32, lora_alpha=64,
                       lora_dropout=.05, warmup_steps=5, weight_decay=.001, scheduler="linear",
                       optimizer="adamw_8bit", logging_steps=10)
            path = root / "c.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(read_config(path)["output_dir"], str((ROOT / "output" / "v3" / "unit-test").resolve()))
            cfg["stt_adapter_for_merge"] = str(root / "persona" / "adapter")
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Forbidden path"):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
