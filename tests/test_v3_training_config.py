import json
import tempfile
import unittest
from pathlib import Path

from training.run_v3 import ROOT, read_config, response_labels, strip_one_bos


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
            cfg = dict(zip(("base_model", "stt_adapter", "train_data", "val_data"), inputs))
            cfg.update(output_dir=str(ROOT / "output" / "v3" / "unit-test"), max_length=2048, epochs=1,
                       learning_rate=2e-5, seed=3407, data_seed=3407, save_steps=200,
                       batch_size=1, gradient_accumulation_steps=4)
            path = root / "c.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(read_config(path)["output_dir"], str((ROOT / "output" / "v3" / "unit-test").resolve()))
            cfg["stt_adapter"] = str(root / "persona" / "adapter")
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Forbidden path"):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
