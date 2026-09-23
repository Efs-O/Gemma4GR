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

    def test_resume_skips_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "done.jsonl"
            row = {"run_key": "stock_q4", "case_id": "1", "settings_fingerprint": "abc"}
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertIn(("stock_q4", "1"), ev.completed_cases(p, "abc"))

    def test_resume_rejects_mismatching_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "done.jsonl"
            row = {"run_key": "stock_q4", "case_id": "1", "settings_fingerprint": "old"}
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale rows from a different configuration"):
                ev.completed_cases(p, "new")

    def test_fingerprint_covers_all_run_settings(self):
        settings = ev.run_settings(Path("model.gguf"), Path("mmproj.gguf"), "text")
        expected = ev.settings_fingerprint(settings)
        self.assertEqual(ev.settings_fingerprint(dict(settings)), expected)
        for field, value in (("server_command", "other"), ("cache_prompt", True), ("slots", 2),
                             ("max_tokens", 100), ("temperature", 0.5), ("seed", 1),
                             ("chat_template_kwargs", {"enable_thinking": True})):
            changed = dict(settings)
            changed[field] = value
            self.assertNotEqual(ev.settings_fingerprint(changed), expected)

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
