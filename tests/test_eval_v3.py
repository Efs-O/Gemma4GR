import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import eval_v3 as ev


class MetricsTests(unittest.TestCase):
    def test_chrf_and_f1(self):
        self.assertEqual(ev.chrf("abc", "abc"), 1.0)
        self.assertEqual(ev.token_f1("hello world", "hello world"), 1.0)
        self.assertEqual(ev.token_f1("a b", "c d"), 0.0)

    def test_script_and_garbage_metrics(self):
        self.assertEqual(ev.greek_share("Ελλάδα 123"), 1.0)
        self.assertEqual(ev.metrics("Καλή λέξη.") ["garbage"], False)
        self.assertTrue(ev.garbage_fragments("Μέλτ++){}μπεμι"))


class ParsingAndAudioTests(unittest.TestCase):
    def test_payload_disables_prompt_cache(self):
        payload = ev.request_payload([{"role": "user", "content": "Γεια"}], "text")
        self.assertIs(payload["cache_prompt"], False)

    def test_resume_skip_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "done.jsonl"
            p.write_text(json.dumps({"run_key": "stock_q4", "case_id": "1"}) + "\n", encoding="utf-8")
            self.assertIn(("stock_q4", "1"), ev.completed_cases(p))
            self.assertNotIn(("v2fixed_q4", "1"), ev.completed_cases(p))

    def test_system_prompt_parse(self):
        row = f"<bos><|turn>system\n{ev.SYSTEM}<turn|>\n<|turn>user\nΓεια<turn|>\n<|turn>model\nΧαίρετε<turn|>\n"
        messages, ref = ev.parse_rendered(row)
        self.assertEqual(ref, "Χαίρετε")
        self.assertEqual(messages[-1]["content"], "Γεια")
        with self.assertRaises(ValueError):
            ev.parse_rendered(row.replace(ev.SYSTEM, "wrong"))

    def test_audio_sha(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.wav"
            p.write_bytes(b"test")
            self.assertEqual(ev.verify_audio_sha(p, hashlib.sha256(b"test").hexdigest()), b"test")
            with self.assertRaises(ValueError): ev.audio_bytes(p, "0" * 64)


if __name__ == "__main__":
    unittest.main()
