import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "common.py"
spec = importlib.util.spec_from_file_location("common", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class CommonTests(unittest.TestCase):
    def test_parse_ranked_candidates_sorts_pairs(self):
        ranked = mod.parse_ranked_candidates([[3, 0.1], [5, 2.0], [4, 1.0]], top_k=2)
        self.assertEqual(ranked, [5, 4])

    def test_parse_ranked_candidates_keeps_integer_order(self):
        ranked = mod.parse_ranked_candidates([7, 2, 9], top_k=2)
        self.assertEqual(ranked, [7, 2])

    def test_calculate_metrics(self):
        ranked = {0: [3, 2, 1], 1: [5, 4]}
        qrels = {0: {1}, 1: {4}}
        metrics = mod.calculate_metrics(ranked, qrels, [1, 3])
        self.assertAlmostEqual(metrics["MRR@1"], 0.0)
        self.assertAlmostEqual(metrics["MRR@3"], (1 / 3 + 1 / 2) / 2)
        self.assertAlmostEqual(metrics["Recall@3"], 1.0)

    def test_qrels_from_details_uses_positive_fields(self):
        rows = [
            {
                "search_result_details_with_idx": [
                    {"note_idx": 10, "click": 0, "like": 0},
                    {"note_idx": 11, "click": 0, "like": 1},
                ]
            }
        ]
        self.assertEqual(mod.qrels_from_details(rows, 5, ["click", "like"]), {5: {11}})

    def test_build_note_text_missing_policy(self):
        notes = {1: {"note_title": "title", "note_content": "content"}}
        self.assertEqual(
            mod.build_note_text(1, notes, {}, "summary", "fallback", 0),
            "title\ncontent",
        )
        self.assertIsNone(mod.build_note_text(1, notes, {}, "summary", "drop", 0))
        combined = mod.build_note_text(1, notes, {1: "card"}, "combined", "drop", 0)
        self.assertIn("card", combined)
        self.assertIn("title", combined)

    def test_paper_representations(self):
        notes = {1: {"note_title": "title", "note_content": "content"}}
        cards = {1: "card"}
        self.assertEqual(mod.build_note_text(1, notes, cards, "original", "error", 0), "title\ncontent")
        self.assertEqual(mod.build_note_text(1, notes, cards, "summary", "error", 0), "card")
        self.assertEqual(
            mod.build_note_text(1, notes, cards, "combined", "error", 0),
            "[Notecard]\ncard\n[Original Note]\nTitle: title\nContent: content",
        )

    def test_title_summary_content_head_keeps_title_card_and_content_head(self):
        notes = {1: {"note_title": "title", "note_content": "abcdef"}}
        text = mod.build_note_text(1, notes, {1: "card"}, "title_summary_content_head", "error", 0, 3)
        self.assertIn("title", text)
        self.assertIn("card", text)
        self.assertIn("abc", text)
        self.assertNotIn("def", text)

    def test_title_content_head_does_not_require_notecard(self):
        notes = {1: {"note_title": "title", "note_content": "abcdef"}}
        text = mod.build_note_text(1, notes, {}, "title_content_head", "error", 0, 2)
        self.assertIn("title", text)
        self.assertIn("ab", text)

    def test_qilin_train_groups_use_click_one_positive_and_click_zero_negative(self):
        rows = [
            {
                "query": "q",
                "search_result_details_with_idx": [
                    {"note_idx": 1, "click": 1},
                    {"note_idx": 2, "click": 0},
                ],
            }
        ]
        notes = {
            1: {"note_title": "p", "note_content": ""},
            2: {"note_title": "n", "note_content": ""},
        }
        groups = mod.build_qilin_train_groups(rows, notes, {}, "original", "error", 0, 300, "click_zero", ["click"])
        self.assertEqual(groups[0].positive_ids, [1])
        self.assertEqual(groups[0].negative_ids, [2])

    def test_qilin_official_nonzero_policy_matches_public_source(self):
        rows = [
            {
                "query": "q",
                "search_result_details_with_idx": [
                    {"note_idx": 1, "click": 1},
                    {"note_idx": 2, "click": 0},
                ],
            }
        ]
        notes = {
            1: {"note_title": "p", "note_content": ""},
            2: {"note_title": "n", "note_content": ""},
        }
        groups = mod.build_qilin_train_groups(
            rows, notes, {}, "original", "error", 0, 300, "qilin_official_nonzero", ["click"]
        )
        self.assertEqual(groups[0].positive_ids, [1])
        self.assertEqual(groups[0].negative_ids, [1])


if __name__ == "__main__":
    unittest.main()
