"""Run the September TF-IDF retrieval baseline on the frozen test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.joinpath("data", "processed", "corpus_pages.jsonl")
DEFAULT_QUESTIONS = ROOT.joinpath("data", "processed", "eval_questions.jsonl")
DEFAULT_TEST_IDS = ROOT.joinpath("data", "processed", "test_ids.json")
DEFAULT_OUTPUT = ROOT.joinpath("results", "keyword_baseline.json")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_page_ids(
    question: str,
    vectorizer: TfidfVectorizer,
    corpus_matrix: Any,
    page_ids: list[str],
    k: int,
) -> list[str]:
    scores = (corpus_matrix @ vectorizer.transform([question]).T).toarray().ravel()
    indices = np.argsort(-scores, kind="stable")[:k]
    return [page_ids[index] for index in indices]


def evaluate(
    corpus: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    test_ids: set[str],
    k: int = 5,
) -> dict[str, Any]:
    by_question_id = {row["id"]: row for row in questions}
    unknown_ids = sorted(test_ids.difference(by_question_id))
    if unknown_ids:
        raise ValueError(f"Frozen test IDs missing from questions: {unknown_ids}")

    page_ids = [row["page_id"] for row in corpus]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("Corpus page IDs must be unique")

    gold_page_ids = {
        page_id
        for question_id in test_ids
        for page_id in by_question_id[question_id]["gold_evidence_pages"]
    }
    missing_gold_pages = sorted(gold_page_ids.difference(page_ids))
    if missing_gold_pages:
        raise ValueError(f"Gold evidence pages missing from corpus: {missing_gold_pages}")

    vectorizer = TfidfVectorizer(stop_words="english")
    corpus_matrix = vectorizer.fit_transform([row["text"] for row in corpus])
    results = []
    for question_id in sorted(test_ids):
        question = by_question_id[question_id]
        retrieved_page_ids = rank_page_ids(
            question["question"], vectorizer, corpus_matrix, page_ids, k
        )
        gold = question["gold_evidence_pages"]
        results.append(
            {
                "id": question_id,
                "gold_evidence_pages": gold,
                "retrieved_page_ids": retrieved_page_ids,
                "hit": bool(set(gold).intersection(retrieved_page_ids)),
            }
        )

    hits = sum(result["hit"] for result in results)
    return {
        "method": "tfidf_word_unigram",
        "k": k,
        "corpus_page_count": len(corpus),
        "test_question_count": len(results),
        "hits": hits,
        "hit_at_k": hits/len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    test_ids = set(json.loads(args.test_ids.read_text(encoding="utf-8"))["test_ids"])
    report = evaluate(load_jsonl(args.corpus), load_jsonl(args.questions), test_ids, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    print(f"Hit@{args.k}: {report['hit_at_k']:.1%} ({report['hits']}/{report['test_question_count']})")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
