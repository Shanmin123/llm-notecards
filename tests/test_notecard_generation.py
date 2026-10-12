import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notecard_generation"))

import evaluate_card_quality as quality  # noqa: E402
import generate_notecards as gen  # noqa: E402

CARD = "类目: 穿搭\n关键词: 羽绒服, 短款\n场景: 冬季穿搭\n人群: 小个子女生\n实体: 蕉内\n摘要: 蕉内短款羽绒服"


class DenoiseTests(unittest.TestCase):
    def test_removes_topic_hashtags_markers_and_emoji(self):
        text = "好看#穿搭[话题]#  真的[笑哭R]绝了😂   推荐"
        self.assertEqual(gen.regex_denoise(text), "好看 真的绝了 推荐")

    def test_keeps_plain_hashtags(self):
        self.assertEqual(gen.regex_denoise("#南京国际博览中心# 奶疙瘩"), "#南京国际博览中心# 奶疙瘩")


class CardFormatTests(unittest.TestCase):
    def test_valid_card(self):
        self.assertTrue(gen.is_valid_card(CARD))

    def test_full_width_colons_parse(self):
        self.assertTrue(gen.is_valid_card(CARD.replace(": ", "：")))

    def test_missing_field_is_malformed(self):
        broken = "\n".join(line for line in CARD.splitlines() if not line.startswith("场景"))
        self.assertFalse(gen.is_valid_card(broken))
        self.assertFalse(gen.is_valid_card(None))

    def test_prompt_contains_note(self):
        prompt = gen.build_prompt("标题", "正文")
        self.assertTrue(prompt.endswith("Title: 标题\nBody: 正文\n"))
        self.assertEqual(prompt.count("示例"), 5)


class QualityAuditTests(unittest.TestCase):
    def test_none_marker_counts_as_invalid_field(self):
        cards = {"1": CARD, "2": CARD.replace("实体: 蕉内", "实体: 无")}
        result = quality.audit(cards, ["1", "2"])
        self.assertEqual(result["format_broken"], 0)
        self.assertAlmostEqual(result["field_validity"]["Entities"], 0.5)
        self.assertAlmostEqual(result["fully_complete_rate"], 0.5)

    def test_field_missing_everywhere_is_not_in_schema(self):
        english = "Category: 美食\nKeywords: 汉堡\nAudience: 美食爱好者\nEntities: None\nGist: 汉堡测评"
        result = quality.audit({"1": english}, ["1"])
        self.assertEqual(result["format_broken"], 0)
        self.assertIsNone(result["field_validity"]["Scenario"])
        self.assertAlmostEqual(result["field_validity"]["Entities"], 0.0)


if __name__ == "__main__":
    unittest.main()
