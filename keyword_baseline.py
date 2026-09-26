"""Rank report pages with TF-IDF and BM25, then report retrieval scores."""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
PAGES_FILE = ROOT.joinpath("data", "processed", "corpus_pages.jsonl")
QUESTIONS_FILE = ROOT.joinpath("data", "processed", "eval_questions.jsonl")
TEST_IDS_FILE = ROOT.joinpath("data", "processed", "test_ids.json")
REPORT_FILE = ROOT.joinpath("results", "keyword_baseline.json")
BM25_REPORT_FILE = ROOT.joinpath("results", "bm25_baseline.json")
TOP_K = 5
MRR_K = 10


def read_jsonl(path):
    """Read nonblank JSON objects from a UTF-8 JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        Parsed objects in file order.
    """
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rank_page_ids(question, pages, k=TOP_K):
    """Return the top ``k`` page IDs for one question.

    Args:
        question: Text to search for.
        pages: Page records with ``page_id`` and ``text`` fields.
        k: Maximum number of IDs to return.

    Returns:
        Page IDs ranked by similarity, highest first.
    """
    # TF-IDF gives distinctive words more weight than words found everywhere.
    # scikit-learn handles turning page and question text into word weights.
    vectorizer = TfidfVectorizer(stop_words="english")
    page_texts = []
    for page in pages:
        page_texts.append(page["text"])
    page_vectors = vectorizer.fit_transform(page_texts)
    question_vector = vectorizer.transform([question])

    # This named library function compares the question with every page.
    scores = cosine_similarity(question_vector, page_vectors)[0]
    # zip pairs each page with its score at the same position.
    scored_pages = []
    for score, page in zip(scores, pages):
        scored_pages.append((score, page["page_id"]))
    # Sort by score, the first tuple item. Ties keep the original page order.
    scored_pages.sort(key=lambda scored_page: scored_page[0], reverse=True)

    ranked_page_ids = []
    for score, page_id in scored_pages[:k]:
        ranked_page_ids.append(page_id)
    return ranked_page_ids


def rank_page_ids_bm25(question, pages, k=TOP_K):
    """Return page IDs ranked by Okapi BM25.

    Args:
        question: Text to search for.
        pages: Page records with ``page_id`` and ``text`` fields.
        k: Maximum number of IDs to return.

    Returns:
        Page IDs ranked by BM25 score, highest first.
    """
    # Share TF-IDF's lowercase, word, and stop-word processing for comparison.
    analyze = TfidfVectorizer(stop_words="english").build_analyzer()
    page_words = []
    for page in pages:
        page_words.append(analyze(page["text"]))
    bm25 = BM25Okapi(page_words)
    scores = bm25.get_scores(analyze(question))
    # zip pairs each page with its score at the same position.
    scored_pages = []
    for score, page in zip(scores, pages):
        scored_pages.append((score, page["page_id"]))
    # Sort by score, the first tuple item. Ties keep the original page order.
    scored_pages.sort(key=lambda scored_page: scored_page[0], reverse=True)

    ranked_page_ids = []
    for score, page_id in scored_pages[:k]:
        ranked_page_ids.append(page_id)
    return ranked_page_ids


def evaluate(pages, questions, test_ids, k=TOP_K, method="tfidf"):
    """Evaluate page search on the fixed question IDs.

    Args:
        pages: Page records with IDs and text.
        questions: Records with IDs, question text, and known source page IDs.
        test_ids: Question IDs to evaluate.
        k: Number of retrieved pages to check for a hit.
        method: ``"tfidf"`` or ``"bm25"``.

    Returns:
        Hit@1, Hit@k, and MRR@10, plus each question's expected and retrieved
        page IDs, full retrieved page records, and hit status.

    Raises:
        ValueError: The test set is empty, an ID is missing, or a page ID
            repeats.
    """
    if not test_ids:
        raise ValueError("The test question list is empty")
    if method not in {"tfidf", "bm25"}:
        raise ValueError("Method must be 'tfidf' or 'bm25'")

    questions_by_id = {}
    for question in questions:
        questions_by_id[question["id"]] = question

    pages_by_id = {}
    for page in pages:
        pages_by_id[page["page_id"]] = page
    if len(pages_by_id) != len(pages):
        raise ValueError("Each report page needs a unique page ID")

    results = []
    for question_id in sorted(test_ids):
        if question_id not in questions_by_id:
            raise ValueError(f"Test question is missing: {question_id}")

        question = questions_by_id[question_id]
        correct_page_ids = question["gold_evidence_pages"]
        for page_id in correct_page_ids:
            if page_id not in pages_by_id:
                raise ValueError(f"Correct page is missing for: {question_id}")

        # MRR@10 needs ten ranked pages, while the saved evidence stays at top five.
        rank_count = max(k, MRR_K)
        if method == "bm25":
            ranked_page_ids = rank_page_ids_bm25(question["question"], pages, rank_count)
        else:
            ranked_page_ids = rank_page_ids(question["question"], pages, rank_count)

        retrieved_page_ids = ranked_page_ids[:k]
        first_correct_rank_at_10 = None
        # Start at one because search results are described as first, second, etc.
        for rank, page_id in enumerate(ranked_page_ids[:MRR_K], start=1):
            if page_id in correct_page_ids:
                first_correct_rank_at_10 = rank
                break
        if first_correct_rank_at_10 is None:
            reciprocal_rank_at_10 = 0.0
        else:
            reciprocal_rank_at_10 = 1 / first_correct_rank_at_10

        hit_at_1 = False
        if retrieved_page_ids:
            hit_at_1 = retrieved_page_ids[0] in correct_page_ids

        hit_at_k = False
        for page_id in retrieved_page_ids:
            if page_id in correct_page_ids:
                hit_at_k = True
                break

        retrieved_pages = []
        for page_id in retrieved_page_ids:
            retrieved_pages.append(pages_by_id[page_id])

        results.append({
            "id": question_id,
            "gold_evidence_pages": correct_page_ids,
            "retrieved_page_ids": retrieved_page_ids,
            "retrieved_pages": retrieved_pages,
            "hit_at_1": hit_at_1,
            "hit": hit_at_k,
            "first_correct_rank_at_10": first_correct_rank_at_10,
            "reciprocal_rank_at_10": reciprocal_rank_at_10,
        })

    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_rank_total = 0.0
    for result in results:
        if result["hit_at_1"]:
            hits_at_1 += 1
        if result["hit"]:
            hits_at_k += 1
        reciprocal_rank_total += result["reciprocal_rank_at_10"]
    mrr_at_10 = reciprocal_rank_total / len(results)
    return {
        "method": method,
        "k": k,
        "corpus_page_count": len(pages),
        "test_question_count": len(results),
        "hits_at_1": hits_at_1,
        "hit_at_1": hits_at_1 / len(results),
        "hits": hits_at_k,
        "hit_at_k": hits_at_k / len(results),
        "mrr_at_10": mrr_at_10,
        "results": results,
    }


def main():
    """Run both keyword searches and save their reports.

    Reads the prepared pages, questions, and test IDs. Writes one report per
    method and prints Hit@1, Hit@``TOP_K``, and MRR@10 for each.
    """
    pages = read_jsonl(PAGES_FILE)
    questions = read_jsonl(QUESTIONS_FILE)
    saved_split = json.loads(TEST_IDS_FILE.read_text(encoding="utf-8"))
    test_ids = set(saved_split["test_ids"])
    for method, report_path in (("tfidf", REPORT_FILE), ("bm25", BM25_REPORT_FILE)):
        report = evaluate(pages, questions, test_ids, method=method)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        label = "TF-IDF" if method == "tfidf" else "BM25"
        print(f"{label} Hit@1: {report['hit_at_1']:.1%} ({report['hits_at_1']}/{report['test_question_count']})")
        print(f"{label} Hit@{TOP_K}: {report['hit_at_k']:.1%} ({report['hits']}/{report['test_question_count']})")
        print(f"{label} MRR@{MRR_K}: {report['mrr_at_10']:.3f}")
        print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
