"""Check keyword search with two made-up pages and one question.

The input question asks about assets. The expected output is the assets page
as the top result and a successful Hit@1 score.
"""

import unittest

from keyword_baseline import evaluate


class KeywordBaselineTest(unittest.TestCase):
    def test_reports_a_hit_when_the_gold_page_is_ranked_first(self) -> None:
        """Input: two pages and one question. Output: the assets page is first."""
        corpus = [
            {"page_id": "assets", "text": "Total assets were 100 dollars."},
            {"page_id": "income", "text": "Net income was 5 dollars."},
        ]
        questions = [
            {
                "id": "q-assets",
                "question": "What were total assets?",
                "gold_evidence_pages": ["assets"],
            }
        ]

        report = evaluate(corpus, questions, {"q-assets"}, k=1)

        self.assertEqual(report["hits"], 1)
        self.assertEqual(report["hit_at_k"], 1.0)
        self.assertEqual(report["results"][0]["retrieved_page_ids"], ["assets"])


if __name__ == "__main__":
    unittest.main()
