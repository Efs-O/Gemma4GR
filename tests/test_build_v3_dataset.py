import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v3_dataset import TOKENS, flags, norm, near, parse, row, shingles


def render(user, answer, system="Απάντησε στα Ελληνικά."):
    return f"<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n{answer}<turn|>\n"


class BuildDatasetTests(unittest.TestCase):
    def test_template_and_failure(self):
        self.assertEqual(parse(render("Ερώτηση;", "Αυτή είναι απάντηση."))[1:], ("Ερώτηση;", "Αυτή είναι απάντηση."))
        self.assertIsNone(row('{"text":"bad template"}', 1))

    def test_nfc_normalization_helper(self):
        self.assertEqual(norm("καφε\u0301ς   καλός"), "καφές καλός")
        from scripts.build_v3_dataset import parse
        normalized = render("καφε\u0301ς", "Αυτή είναι μια αρκετά μεγάλη απάντηση.")
        system, user, answer = parse(normalized)
        normalized_render = render(system, norm(user), norm(answer))
        self.assertIn("καφές", normalized_render)
        self.assertNotIn("ε\u0301", normalized_render)

    def test_whitelist(self):
        from collections import Counter
        hits = Counter()
        result, ratio = flags("Πόσο κάνει 5 km;", "Το UNESCO είναι γνωστό.", hits)
        self.assertIn("km", hits)
        self.assertIn("UNESCO", hits)
        self.assertNotIn("latin_heavy", result)

    def test_drop_categories(self):
        self.assertIn("non_greek_script", flags("Τι λέει 标志;", "Αυτή είναι απάντηση.", {})[0])
        self.assertIn("latin_heavy", flags("Explain please", "This is a sufficiently long answer.", {})[0])
        self.assertIn("garbage_in_word", flags("Τι σημαίνει;", "Μέλτ++){}μπεμι", {})[0])
        self.assertIn("mojibake", flags("Τι σημαίνει;", "Απάντηση με � σύμβολο.", {})[0])
        self.assertIn("empty_or_truncated", flags("Τι σημαίνει;", "ΟΚ", {})[0])

    def test_dedup_and_near_dup(self):
        a = {"user":"Τι είναι το μελτέμι;", "answer":"Το μελτέμι είναι ένας εποχικός άνεμος που φυσά το καλοκαίρι."}
        b = {"user":"Τι είναι το μελτέμι;", "answer":"Το μελτέμι είναι ένας εποχικός άνεμος που φυσά το καλοκαίρι."}
        c = {"user":"Τι είναι το μελτέμι;", "answer":"Το μελτέμι είναι ένας εποχικός άνεμος που φυσάει το καλοκαίρι."}
        self.assertEqual((a["user"], a["answer"]), (b["user"], b["answer"]))
        self.assertTrue(near(a, c))
        self.assertTrue(shingles("abcdef"))

    def test_near_uses_shingle_set_bound_not_string_length(self):
        sentence = "Η πολιτισμική κληρονομιά διατηρείται μέσα στους αιώνες."
        a = {"user": "Ποια κληρονομιά;", "answer": sentence}
        b = {"user": "Ποια κληρονομιά;", "answer": (sentence + " ") * 2}
        self.assertGreater(len(b["answer"]), len(a["answer"]) * 1.15)
        self.assertGreaterEqual(len(shingles(norm(a["user"] + " " + a["answer"])) & shingles(norm(b["user"] + " " + b["answer"]))) /
                                len(shingles(norm(a["user"] + " " + a["answer"])) | shingles(norm(b["user"] + " " + b["answer"]))), .85)
        self.assertTrue(near(a, b))

    def test_jsonl_byte_stability(self):
        from scripts.build_v3_dataset import write
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rows.jsonl"
            write(p, [{"text":render("Ερώτηση;", "Αυτή είναι απάντηση.")}])
            first = p.read_bytes()
            write(p, [json.loads(first.decode().splitlines()[0])])
            self.assertEqual(first, p.read_bytes())


if __name__ == "__main__":
    unittest.main()

