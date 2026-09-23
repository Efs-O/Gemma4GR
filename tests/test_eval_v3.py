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


class SummaryTests(unittest.TestCase):
    def test_every_variant_is_paired_with_its_stock_quant(self):
        from unittest.mock import patch

        def row(case_id, chrf):
            return {"case_id": case_id, "metrics": {"chrf": chrf, "garbage": False, "length_chars": 3}, "stop_failure": False}

        with tempfile.TemporaryDirectory() as d, patch.object(ev, "OUT", Path(d)):
            runs = {"stock_Q4_K_M": [0.5, 0.5], "v2fixed_Q4_K_M": [0.5, 0.5], "v3_Q4_K_M": [0.6, 0.4],
                    "v3qaonly_Q4_K_M": [0.6, 0.6], "stock_Q4_K_M_debug": [0.1, 0.1]}
            for key, scores in runs.items():
                (Path(d) / key).mkdir()
                (Path(d) / key / "text.jsonl").write_text(
                    "\n".join(json.dumps(row(i, s)) for i, s in enumerate(scores)), encoding="utf-8")
            ev.summarize()
            paired = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))["paired_chrf_vs_stock"]
        self.assertEqual(set(paired), {f"{k}_Q4_K_M/text vs stock_Q4_K_M" for k in ("v2fixed", "v3", "v3qaonly")})
        self.assertEqual(paired["v3_Q4_K_M/text vs stock_Q4_K_M"], {"wins": 1, "ties": 0, "losses": 1})
        self.assertEqual(paired["v3qaonly_Q4_K_M/text vs stock_Q4_K_M"], {"wins": 2, "ties": 0, "losses": 0})
