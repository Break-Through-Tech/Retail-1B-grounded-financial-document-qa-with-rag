import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_configurations import (
    DEFAULT_QUESTIONS,
    DEFAULT_SPLIT,
    RecordingRetriever,
    Retrievers,
    evaluate_answers,
    evaluate_retrieval,
    first_gold_rank,
    grade_answer,
    hit_at_k,
    load_dev_questions,
    load_manual_labels,
    parse_citations,
    reciprocal_rank_at_k,
    reciprocal_rank_fusion,
    score_answer,
    summarize_answers,
)
from dense_index import DenseIndex, build


class FakeModel:
    """Bag-of-words stand-in for a SentenceTransformer, so tests need no download."""

    vocab = ["assets", "income", "revenue", "cash", "debt"]

    def encode(self, texts, normalize_embeddings=True, **_):
        vectors = np.array(
            [[text.lower().count(word) + 0.01 for word in self.vocab] for text in texts],
            dtype="float32",
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def page(doc, num, text):
    return {
        "page_id": f"{doc}_p{num}",
        "doc_name": doc,
        "company": doc.split("_")[0],
        "doc_type": "10k",
        "doc_period": 2020,
        "gics_sector": "Industrials",
        "page_num": num,
        "doc_link": "https://example.com/doc.pdf",
        "text": text,
    }


PAGES = [
    page("ACME_2020_10K", 3, "Total assets were 100 dollars. Assets grew."),
    page("ACME_2020_10K", 9, "Net income and revenue increased. Revenue was 50."),
    page("BOLT_2021_10K", 4, "Cash and cash equivalents fell. Debt rose."),
    page("BOLT_2021_10K", 7, "Long-term debt and debt maturities are listed."),
]

QUESTIONS = [
    {"id": "q1", "question": "What were total assets?", "answer": "$100.00",
     "gold_evidence_pages": ["ACME_2020_10K_p3"], "question_type": "metrics-generated"},
    {"id": "q2", "question": "How much debt is there?", "answer": "Yes, debt rose.",
     "gold_evidence_pages": ["BOLT_2021_10K_p7"], "question_type": "novel-generated"},
    {"id": "q3", "question": "Describe revenue drivers", "answer": "Higher prices and volume.",
     "gold_evidence_pages": ["ACME_2020_10K_p9"], "question_type": "domain-relevant"},
]


class FakeReranker:
    """Mimics HybridRetrieveRerank.search: returns pages with a hybrid_score, and prints."""

    def __init__(self, order, scores=None):
        self.order = order
        self.scores = scores or {}

    def search(self, query, top_res=5):
        print("-> this output should be hidden")
        by_id = {row["page_id"]: row for row in PAGES}
        return [{**by_id[pid], "hybrid_score": self.scores.get(pid, 5.0 - i)} for i, pid in enumerate(self.order[:top_res])]


class FakeGenerator:
    """Mimics QwenGeneratorRetail.generate: retrieve, apply threshold, cite the top page."""

    def __init__(self, retriever, min_evidence_score=-3.0):
        self.docRetriever = retriever
        self.min_evidence_score = min_evidence_score
        self.calls = 0

    def generate(self, query, top_k=5):
        self.calls += 1
        chunks = self.docRetriever.search(query, top_res=top_k)
        if not chunks or chunks[0]["hybrid_score"] < self.min_evidence_score:
            return "I can't answer this question from the available documents."
        top = chunks[0]
        return f"The answer is $100 [{top['doc_name']}, Page {top['page_num']}]."


def make_retrievers(tmp, reranker=None):
    corpus_path = Path(tmp) / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in PAGES), encoding="utf-8")
    build(corpus_path, Path(tmp) / "index", model=FakeModel(), chunk_words=6, overlap_words=2)
    dense = DenseIndex.load(Path(tmp) / "index", model=FakeModel())
    return Retrievers(PAGES, dense_index=dense, reranker=reranker)


class RetrievalMetricTests(unittest.TestCase):
    def test_rank_hit_and_reciprocal_rank(self):
        rank = first_gold_rank(["a", "b", "c"], ["c", "z"])
        self.assertEqual(rank, 3)
        self.assertTrue(hit_at_k(rank, 5))
        self.assertFalse(hit_at_k(rank, 2))
        self.assertAlmostEqual(reciprocal_rank_at_k(rank, 10), 1 / 3)
        self.assertEqual(reciprocal_rank_at_k(None, 10), 0.0)
        self.assertEqual(reciprocal_rank_at_k(11, 10), 0.0)

    def test_reciprocal_rank_fusion_rewards_agreement(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
        self.assertEqual(fused[0], "b")  # ranks 2 and 1 beat ranks 1 and 3
        self.assertEqual(set(fused), {"a", "b", "c", "d"})


class RetrieverTests(unittest.TestCase):
    def test_every_setup_returns_distinct_pages_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = ["BOLT_2021_10K_p7", "BOLT_2021_10K_p4", "ACME_2020_10K_p3", "ACME_2020_10K_p9"]
            retrievers = make_retrievers(tmp, FakeReranker(order))
            for config in ("keyword", "dense", "hybrid", "hybrid_rerank"):
                hits = retrievers.retrieve(config, "debt and cash", 3)
                ids = [hit["page_id"] for hit in hits]
                self.assertEqual(len(ids), len(set(ids)), config)
                self.assertLessEqual(len(ids), 3)
                for hit in hits:
                    self.assertIn("score", hit)
                    self.assertIn("doc_name", hit)
                    self.assertIn("page_num", hit)
            self.assertTrue(retrievers.retrieve("keyword", "debt maturities", 1)[0]["page_id"].startswith("BOLT"))
            self.assertEqual(retrievers.retrieve("hybrid_rerank", "x", 2)[0]["page_id"], order[0])

    def test_keyword_only_needs_no_models(self):
        retrievers = Retrievers(PAGES)
        summary, rows = evaluate_retrieval(retrievers, QUESTIONS, ["keyword"])
        self.assertEqual(summary[0]["questions"], 3)
        self.assertEqual(len(rows), 3)
        self.assertGreaterEqual(summary[0]["hit_at_5"], summary[0]["mrr_at_10"])
        with self.assertRaises(RuntimeError):
            retrievers.retrieve("dense", "x", 1)

    def test_retrieval_metrics_match_hand_computation(self):
        order = ["BOLT_2021_10K_p7", "ACME_2020_10K_p9", "ACME_2020_10K_p3", "BOLT_2021_10K_p4"]
        retrievers = Retrievers(PAGES, reranker=FakeReranker(order))
        summary, _ = evaluate_retrieval(retrievers, QUESTIONS, ["hybrid_rerank"])
        # gold ranks: q1 -> 3, q2 -> 1, q3 -> 2
        self.assertAlmostEqual(summary[0]["hit_at_5"], 1.0)
        self.assertAlmostEqual(summary[0]["mrr_at_10"], (1 / 3 + 1 + 1 / 2) / 3)


class AnswerGradingTests(unittest.TestCase):
    def test_numeric_answers(self):
        self.assertTrue(grade_answer("FY2018 capex was $1,577 million [3M_2018_10K, Page 59].", "$1577.00"))
        self.assertTrue(grade_answer("Capital expenditure was $1.577 billion.", "$1577.00"))
        self.assertTrue(grade_answer("The margin decreased by 1.87%.", "1.9%"))
        self.assertTrue(grade_answer("The ratio is 0.019, i.e. a small share.", "1.9%"))
        self.assertTrue(grade_answer("Purchases of PP&E were (1,577).", "$1577.00"))
        self.assertFalse(grade_answer("The quick ratio is 0.71.", "0.66"))
        self.assertFalse(grade_answer("Capex was $1,373 million.", "$1577.00"))

    def test_page_numbers_and_years_are_not_answers(self):
        # 59 is a page and 2018 a year; neither should count as the answer.
        self.assertFalse(grade_answer("In FY2018 it rose [3M_2018_10K, Page 59].", "59"))
        self.assertFalse(grade_answer("Revenue rose in 2018.", "2018"))

    def test_intermediate_calculation_values_are_ignored(self):
        answer = "Revenue was $500 and R&D was $40.\nR&D / revenue = 40 / 500.\nSo R&D was 8.0% of revenue."
        self.assertTrue(grade_answer(answer, "8.0%"))
        self.assertFalse(grade_answer("Revenue was $500.\nR&D was $40.\nSo the ratio is 12%.", "40"))

    def test_yes_no_and_free_text(self):
        self.assertTrue(grade_answer("Yes. Dividends were raised every year.", "Yes, they increased dividends."))
        self.assertFalse(grade_answer("**No**, it did not.", "Yes"))
        self.assertIsNone(grade_answer("Packaging for food and healthcare.", "Amcor is a packaging leader."))
        self.assertFalse(grade_answer("I can't answer this question from the available documents.", "$5.00"))

    def test_parse_citations(self):
        answer = "A [3M_2018_10K, Page 59] and B [AMCOR_2023_10K, page 7]; not [Note 3]."
        self.assertEqual(parse_citations(answer), ["3M_2018_10K_p59", "AMCOR_2023_10K_p7"])

    def test_score_and_summarize(self):
        question = QUESTIONS[0]
        good = score_answer("Assets were $100 [ACME_2020_10K, Page 3].", question, ["ACME_2020_10K_p3"])
        self.assertEqual((good["correct"], good["cites_gold_page"], good["unsupported"]), (True, True, False))
        wrong_page = score_answer("Assets were $100 [ACME_2020_10K, Page 9].", question, ["ACME_2020_10K_p9"])
        self.assertFalse(wrong_page["cites_gold_page"])
        self.assertFalse(wrong_page["unsupported"])  # cites a passage it saw, answer right
        no_cite = score_answer("Assets were $100.", question, ["ACME_2020_10K_p3"])
        self.assertTrue(no_cite["unsupported"])
        wrong = score_answer("Assets were $70 [ACME_2020_10K, Page 3].", question, ["ACME_2020_10K_p3"])
        self.assertTrue(wrong["unsupported"])
        declined = score_answer("I can't answer this question from the available documents.", question, [])
        self.assertFalse(declined["unsupported"])
        self.assertFalse(declined["correct"])
        labeled = score_answer("Something else.", QUESTIONS[2], [], manual_label=True)
        self.assertEqual(labeled["graded_by"], "person")

        rows = [good, wrong_page, no_cite, wrong, declined]
        summary = summarize_answers("dense", rows)
        self.assertAlmostEqual(summary["answer_accuracy"], 3 / 5)
        self.assertAlmostEqual(summary["citation_correctness"], 2 / 4)  # declined answers excluded
        self.assertAlmostEqual(summary["unsupported_answer_rate"], 2 / 5)
        self.assertAlmostEqual(summary["decline_rate"], 1 / 5)
        self.assertTrue(math.isnan(summarize_answers("x", [score_answer("Text.", QUESTIONS[2], [])])["answer_accuracy"]))


class AnswerLoopTests(unittest.TestCase):
    def test_generates_once_per_question_resumes_and_restores_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = ["ACME_2020_10K_p3", "BOLT_2021_10K_p7", "ACME_2020_10K_p9", "BOLT_2021_10K_p4"]
            original = FakeReranker(order, scores={pid: -10.0 for pid in order})  # below threshold
            retrievers = make_retrievers(tmp, original)
            generator = FakeGenerator(original)
            out = Path(tmp)
            configs = ["keyword", "hybrid_rerank"]

            summary = evaluate_answers(generator, retrievers, QUESTIONS, configs, out, {})
            self.assertEqual(generator.calls, 6)
            self.assertIs(generator.docRetriever, original)
            self.assertEqual(generator.min_evidence_score, -3.0)
            by_config = {row["config"]: row for row in summary}
            # Threshold applies only to the reranked setup, whose scores are all below it.
            self.assertEqual(by_config["hybrid_rerank"]["decline_rate"], 1.0)
            self.assertEqual(by_config["keyword"]["decline_rate"], 0.0)

            graded = [json.loads(line) for line in (out / "answers_keyword_graded.jsonl").open()]
            self.assertEqual(len(graded), 3)
            self.assertTrue(all(row["retrieved_page_ids"] for row in graded))

            # Label a needs-review answer, then re-run: nothing regenerates and the label sticks.
            review = (out / "answers_to_review.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(review[0].startswith("config,id"))
            labeled = [line + "1" if line.startswith("keyword,q3,") else line for line in review]
            (out / "answers_to_review.csv").write_text("\n".join(labeled) + "\n", encoding="utf-8")
            labels = load_manual_labels(out / "answers_to_review.csv")
            self.assertEqual(labels, {("keyword", "q3"): True})

            evaluate_answers(generator, retrievers, QUESTIONS, configs, out, labels)
            self.assertEqual(generator.calls, 6)
            self.assertEqual(load_manual_labels(out / "answers_to_review.csv"), labels)

    def test_recording_retriever_exposes_hybrid_score(self):
        adapter = RecordingRetriever(Retrievers(PAGES), "keyword")
        hits = adapter.search("total assets", top_res=2)
        self.assertEqual(adapter.last_results, hits)
        self.assertTrue(all("hybrid_score" in hit for hit in hits))


class DevSplitTests(unittest.TestCase):
    def test_dev_split_never_includes_test_questions(self):
        dev = load_dev_questions(DEFAULT_QUESTIONS, DEFAULT_SPLIT)
        test_ids = set(json.loads(DEFAULT_SPLIT.read_text(encoding="utf-8"))["test_ids"])
        self.assertEqual(len(dev), 118)
        self.assertFalse({row["id"] for row in dev} & test_ids)


if __name__ == "__main__":
    unittest.main()
