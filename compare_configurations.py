"""Task #5: compare keyword, dense, hybrid, and reranked setups on the dev split.

Metrics: Hit@5, MRR@10, answer accuracy, citation correctness, unsupported-answer rate.
Results and charts are saved to results/config_comparison/.

    python3 compare_configurations.py --skip-answers   # retrieval metrics only
    python3 compare_configurations.py                  # everything (needs a GPU)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

from sklearn.feature_extraction.text import TfidfVectorizer

from dense_index import DEFAULT_INDEX_DIR, load_jsonl
from keyword_baseline import rank_page_ids


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.joinpath("data", "processed", "corpus_pages.jsonl")
DEFAULT_QUESTIONS = ROOT.joinpath("data", "processed", "eval_questions.jsonl")
DEFAULT_SPLIT = ROOT.joinpath("data", "processed", "test_ids.json")
DEFAULT_OUTPUT_DIR = ROOT.joinpath("results", "config_comparison")

CONFIGS = ("keyword", "dense", "hybrid", "hybrid_rerank")
CONFIG_LABELS = {
    "keyword": "Keyword (TF-IDF)",
    "dense": "Dense (FAISS)",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}
HIT_K = 5
MRR_K = 10
CANDIDATES_PER_RETRIEVER = 20  # same candidate pool size as HybridRetrieveRerank
RRF_K = 60  # standard reciprocal rank fusion constant
ANSWER_TOP_K = 5  # passages given to the generator, same default as Task #3
SEED = 42

# Matches the insufficient-evidence reply from answer_generation.py.
DECLINE_PATTERN = re.compile(r"can.?t answer this question from the available documents", re.I)
CITATION_PATTERN = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*Page\s+(\d+)\s*\]", re.I)
NUMBER_PATTERN = re.compile(r"-?\(?\$?\s?\d[\d,]*(?:\.\d+)?\)?\s?%?")
NUMERIC_ANSWER_PATTERN = re.compile(r"^\s*-?\$?\s?-?[\d,]*\.?\d+\s?%?\s*$")
YEAR_PATTERN = re.compile(r"(?<![$\d.,])\b(?:FY\s?)?(?:19|20)\d{2}\b(?![.,]\d|\s?%)", re.I)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dev_questions(questions_path: Path, split_path: Path) -> list[dict[str, Any]]:
    """Return the dev-split questions, sorted by id. Fails if any id overlaps the test split."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dev_ids, test_ids = set(split["dev_ids"]), set(split["test_ids"])
    if dev_ids & test_ids:
        raise ValueError(f"Dev and test splits overlap: {sorted(dev_ids & test_ids)}")
    by_id = {row["id"]: row for row in load_jsonl(questions_path)}
    missing = sorted(dev_ids.difference(by_id))
    if missing:
        raise ValueError(f"Dev IDs missing from questions: {missing}")
    return [by_id[question_id] for question_id in sorted(dev_ids)]


# ---------------------------------------------------------------------------
# Retrieval setups
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Merge ranked id lists: each id scores sum(1 / (k + rank)). Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


class Retrievers:
    """The four setups behind one interface: ``retrieve(config, query, k)`` -> list of pages.

    Each returned page is a dict with at least ``page_id``, ``doc_name``, ``page_num``,
    ``text`` and ``score``. Heavy models are passed in, so tests can use fakes.
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        dense_index: Any = None,
        reranker: Any = None,
    ) -> None:
        self.corpus = corpus
        self.page_ids = [row["page_id"] for row in corpus]
        if len(self.page_ids) != len(set(self.page_ids)):
            raise ValueError("Corpus page IDs must be unique")
        self.by_page_id = {row["page_id"]: row for row in corpus}
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.corpus_matrix = self.vectorizer.fit_transform([row["text"] for row in corpus])
        self.dense_index = dense_index
        self.reranker = reranker

    def _page(self, page_id: str, score: float) -> dict[str, Any]:
        return {**self.by_page_id[page_id], "score": float(score)}

    def keyword(self, query: str, k: int) -> list[dict[str, Any]]:
        ids = rank_page_ids(query, self.vectorizer, self.corpus_matrix, self.page_ids, k)
        return [self._page(page_id, 1.0 / rank) for rank, page_id in enumerate(ids, start=1)]

    def dense(self, query: str, k: int) -> list[dict[str, Any]]:
        if self.dense_index is None:
            raise RuntimeError("Dense setups need a DenseIndex; build it with dense_index.py")
        return [dict(hit) for hit in self.dense_index.search_pages(query, k)]

    def hybrid(self, query: str, k: int) -> list[dict[str, Any]]:
        n = CANDIDATES_PER_RETRIEVER
        keyword_hits = self.keyword(query, n)
        dense_hits = self.dense(query, n)
        fused = reciprocal_rank_fusion(
            [[hit["page_id"] for hit in keyword_hits], [hit["page_id"] for hit in dense_hits]]
        )
        # Like HybridRetrieveRerank, prefer the dense passage text when both found a page.
        best_text = {hit["page_id"]: hit for hit in keyword_hits + dense_hits}
        results = []
        for rank, page_id in enumerate(fused[:k], start=1):
            results.append({**best_text[page_id], "score": 1.0 / rank})
        return results

    def hybrid_rerank(self, query: str, k: int) -> list[dict[str, Any]]:
        if self.reranker is None:
            raise RuntimeError("hybrid_rerank needs a HybridRetrieveRerank instance")
        with contextlib.redirect_stdout(io.StringIO()):  # its search() prints progress
            hits = self.reranker.search(query, top_res=k)
        return [{**hit, "score": float(hit["hybrid_score"])} for hit in hits]

    def retrieve(self, config: str, query: str, k: int) -> list[dict[str, Any]]:
        if config not in CONFIGS:
            raise ValueError(f"Unknown config {config!r}; choose from {CONFIGS}")
        return getattr(self, config)(query, k)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def first_gold_rank(retrieved_page_ids: list[str], gold_page_ids: list[str]) -> int | None:
    """1-based rank of the first gold page, or None if no gold page was retrieved."""
    gold = set(gold_page_ids)
    for rank, page_id in enumerate(retrieved_page_ids, start=1):
        if page_id in gold:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank_at_k(rank: int | None, k: int) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def evaluate_retrieval(
    retrievers: Retrievers,
    questions: list[dict[str, Any]],
    configs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (summary rows, per-question rows) for Hit@5, MRR@10 and latency."""
    depth = max(HIT_K, MRR_K)
    summary, per_question = [], []
    for config in configs:
        hits, reciprocal_ranks, latencies = 0, [], []
        for question in questions:
            start = time.perf_counter()
            retrieved = retrievers.retrieve(config, question["question"], depth)
            latencies.append((time.perf_counter() - start) * 1000)
            retrieved_ids = [hit["page_id"] for hit in retrieved]
            rank = first_gold_rank(retrieved_ids, question["gold_evidence_pages"])
            hits += hit_at_k(rank, HIT_K)
            reciprocal_ranks.append(reciprocal_rank_at_k(rank, MRR_K))
            per_question.append(
                {
                    "config": config,
                    "id": question["id"],
                    "question_type": question.get("question_type"),
                    "gold_evidence_pages": question["gold_evidence_pages"],
                    "retrieved_page_ids": retrieved_ids,
                    "first_gold_rank": rank,
                    f"hit_at_{HIT_K}": hit_at_k(rank, HIT_K),
                    "latency_ms": round(latencies[-1], 2),
                }
            )
        n = len(questions)
        summary.append(
            {
                "config": config,
                "questions": n,
                f"hit_at_{HIT_K}": hits / n,
                f"mrr_at_{MRR_K}": sum(reciprocal_ranks) / n,
                "mean_latency_ms": sum(latencies) / n,
            }
        )
        print(
            f"{CONFIG_LABELS[config]:<18} Hit@{HIT_K} {hits / n:6.1%}   "
            f"MRR@{MRR_K} {sum(reciprocal_ranks) / n:.3f}   "
            f"{sum(latencies) / n:7.1f} ms/question"
        )
    return summary, per_question


# ---------------------------------------------------------------------------
# Answer grading
# ---------------------------------------------------------------------------

def is_decline(answer: str) -> bool:
    return bool(DECLINE_PATTERN.search(answer))


def parse_citations(answer: str) -> list[str]:
    """Turn every [DocName, Page X] citation into a corpus page_id like DocName_pX."""
    return [f"{doc.strip()}_p{int(page)}" for doc, page in CITATION_PATTERN.findall(answer)]


def parse_number(token: str) -> float | None:
    """Parse '$1,577.00', '(370)', '-4.2%' and similar into a float (percent sign dropped)."""
    negative = "(" in token and ")" in token or token.strip().startswith("-")
    digits = re.sub(r"[^\d.]", "", token)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def decimals(token: str) -> int:
    match = re.search(r"\.(\d+)", token)
    return len(match.group(1)) if match else 0


def candidate_numbers(answer: str) -> list[float]:
    """Numbers stated in the first and last sentences that contain numbers.

    Citations and stand-alone years are removed first, so page numbers and fiscal
    years are not mistaken for the answer. Only the first and last numeric
    sentences are used because a worked calculation lists many intermediate values;
    the model's answer is normally stated at the start or at the end.
    """
    text = CITATION_PATTERN.sub(" ", answer)
    text = YEAR_PATTERN.sub(" ", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if NUMBER_PATTERN.search(s)]
    if not sentences:
        return []
    chosen = {sentences[0], sentences[-1]}
    values = []
    for sentence in chosen:
        for token in NUMBER_PATTERN.findall(sentence):
            value = parse_number(token)
            if value is not None:
                values.append(value)
    return values


def numbers_match(predicted: float, gold: float, gold_decimals: int) -> bool:
    """Within 1% of gold, or equal after rounding to gold's decimal places."""
    rounding = 0.5 * 10 ** (-gold_decimals)
    return abs(predicted - gold) <= max(0.01 * abs(gold), rounding) + 1e-9


def grade_answer(answer: str, gold_answer: str) -> bool | None:
    """True/False when the answer can be graded automatically, else None (needs a person).

    Numeric gold answers ('$1577.00', '24.26', '1.9%') are matched against numbers in
    the answer, allowing for units written differently (millions vs billions, 0.019
    vs 1.9%) and for sign written as a minus or parentheses. Gold answers that start
    with Yes or No are matched against the first word of the answer. A declined
    answer is always wrong, because every FinanceBench question is answerable.
    """
    if is_decline(answer):
        return False
    gold = gold_answer.strip()
    if NUMERIC_ANSWER_PATTERN.match(gold):
        gold_value = parse_number(gold)
        if gold_value is None:
            return None
        places = decimals(gold)
        scales = (1, 1000, 0.001, 100, 0.01)
        return any(
            numbers_match(value * scale, gold_value, places) or numbers_match(-value * scale, gold_value, places)
            for value in candidate_numbers(answer)
            for scale in scales
        )
    gold_word = re.match(r"\W*(yes|no)\b", gold, re.I)
    answer_word = re.match(r"[\W_]*(yes|no)\b", answer, re.I)
    if gold_word and answer_word:
        return gold_word.group(1).lower() == answer_word.group(1).lower()
    return None


def score_answer(
    answer: str,
    question: dict[str, Any],
    retrieved_page_ids: list[str],
    manual_label: bool | None = None,
) -> dict[str, Any]:
    """Grade one generated answer. Returns the per-question fields saved to answers_<setup>.jsonl."""
    declined = is_decline(answer)
    cited = parse_citations(answer)
    gold = set(question["gold_evidence_pages"])
    retrieved = set(retrieved_page_ids)
    correct = manual_label if manual_label is not None else grade_answer(answer, question["answer"])
    cites_gold = bool(gold.intersection(cited))
    cites_retrieved = bool(retrieved.intersection(cited))
    unsupported = (not declined) and (not cites_retrieved or correct is False)
    return {
        "declined": declined,
        "correct": correct,
        "graded_by": "person" if manual_label is not None else ("auto" if correct is not None else "needs_review"),
        "cited_page_ids": cited,
        "cites_gold_page": cites_gold,
        "cites_retrieved_page": cites_retrieved,
        "unsupported": unsupported,
    }


def summarize_answers(config: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    answered = [row for row in rows if not row["declined"]]
    graded = [row for row in rows if row["correct"] is not None]
    correct = sum(bool(row["correct"]) for row in graded)
    return {
        "config": config,
        "questions": n,
        "graded_questions": len(graded),
        "needs_review": n - len(graded),
        # Accuracy over the questions that have a grade; see needs_review for the rest.
        "answer_accuracy": correct / len(graded) if graded else math.nan,
        "citation_correctness": (
            sum(row["cites_gold_page"] for row in answered) / len(answered) if answered else math.nan
        ),
        "unsupported_answer_rate": sum(row["unsupported"] for row in rows) / n if n else math.nan,
        "decline_rate": (n - len(answered)) / n if n else math.nan,
    }


def load_manual_labels(path: Path | None) -> dict[tuple[str, str], bool]:
    """Read answers_to_review.csv after a person has filled the ``correct`` column with 1 or 0."""
    if path is None:
        return {}
    labels = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("correct") or "").strip().lower()
            if value in {"1", "true", "yes", "y"}:
                labels[(row["config"], row["id"])] = True
            elif value in {"0", "false", "no", "n"}:
                labels[(row["config"], row["id"])] = False
    return labels


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class RecordingRetriever:
    """Stands in for ``QwenGeneratorRetail.docRetriever`` so one generator serves every setup.

    QwenGeneratorRetail.generate() calls ``docRetriever.search(query, top_res=...)`` and
    checks ``hybrid_score`` against its evidence threshold. This adapter returns the
    chosen setup's pages, copies their score into ``hybrid_score``, and remembers what it
    returned so citations can be checked against the passages the model actually saw.
    """

    def __init__(self, retrievers: Retrievers, config: str) -> None:
        self.retrievers = retrievers
        self.config = config
        self.last_results: list[dict[str, Any]] = []

    def search(self, query: str, top_res: int = 5) -> list[dict[str, Any]]:
        results = self.retrievers.retrieve(self.config, query, top_res)
        self.last_results = [{**hit, "hybrid_score": hit["score"]} for hit in results]
        return self.last_results


def evaluate_answers(
    generator: Any,
    retrievers: Retrievers,
    questions: list[dict[str, Any]],
    configs: list[str],
    output_dir: Path,
    manual_labels: dict[tuple[str, str], bool],
    seed_fn: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate and grade an answer per question per setup. Resumes from answers_<setup>.jsonl.

    The Task #4 evidence threshold is a cross-encoder score, so it only means
    something for ``hybrid_rerank``. For the other setups it is switched off, so
    those setups always answer; compare their decline rate with that in mind.
    """
    original_retriever = generator.docRetriever
    original_threshold = generator.min_evidence_score
    summary = []
    try:
        for config in configs:
            path = output_dir / f"answers_{config}.jsonl"
            done = {row["id"]: row for row in load_jsonl(path)} if path.exists() else {}
            adapter = RecordingRetriever(retrievers, config)
            generator.docRetriever = adapter
            generator.min_evidence_score = original_threshold if config == "hybrid_rerank" else -math.inf
            rows = []
            with path.open("a", encoding="utf-8") as handle:
                for number, question in enumerate(questions, start=1):
                    if question["id"] in done:
                        saved = done[question["id"]]
                        answer, retrieved_ids = saved["answer"], saved["retrieved_page_ids"]
                    else:
                        if seed_fn is not None:
                            seed_fn(SEED)
                        with contextlib.redirect_stdout(io.StringIO()):
                            answer = generator.generate(question["question"], top_k=ANSWER_TOP_K)
                        retrieved_ids = [hit["page_id"] for hit in adapter.last_results]
                        handle.write(
                            json.dumps(
                                {"id": question["id"], "answer": answer, "retrieved_page_ids": retrieved_ids},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                        print(f"  {CONFIG_LABELS[config]}: {number}/{len(questions)}", end="\r")
                    label = manual_labels.get((config, question["id"]))
                    rows.append(
                        {
                            "config": config,
                            "id": question["id"],
                            "question": question["question"],
                            "gold_answer": question["answer"],
                            "gold_evidence_pages": question["gold_evidence_pages"],
                            "answer": answer,
                            "retrieved_page_ids": retrieved_ids,
                            **score_answer(answer, question, retrieved_ids, label),
                        }
                    )
            write_jsonl(output_dir / f"answers_{config}_graded.jsonl", rows)
            result = summarize_answers(config, rows)
            summary.append(result)
            print(
                f"{CONFIG_LABELS[config]:<18} accuracy {result['answer_accuracy']:6.1%} "
                f"({result['graded_questions']} graded)   citations {result['citation_correctness']:6.1%}   "
                f"unsupported {result['unsupported_answer_rate']:6.1%}   declined {result['decline_rate']:6.1%}"
            )
            write_review_csv(output_dir / "answers_to_review.csv", rows, append=config != configs[0])
    finally:
        generator.docRetriever = original_retriever
        generator.min_evidence_score = original_threshold
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def without_nan(value: Any) -> Any:
    """Replace NaN with None so the JSON output stays valid."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: without_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_nan(item) for item in value]
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_review_csv(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    """Answers a person grades: blank ``correct`` if not yet labeled, else the saved 1/0.

    Already-labeled rows stay in the file, so re-running with
    ``--manual-labels <this file>`` never loses earlier labels.
    """
    fields = ["config", "id", "question", "gold_answer", "answer", "correct"]
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for row in rows:
            if row["graded_by"] in {"needs_review", "person"}:
                label = "" if row["correct"] is None else str(int(row["correct"]))
                writer.writerow({field: row.get(field, "") for field in fields[:-1]} | {"correct": label})


SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # one fixed color per setup


def plot_metrics(summary: list[dict[str, Any]], metrics: dict[str, str], title: str, path: Path) -> None:
    """Grouped bar chart: one group per metric, one bar per setup (same color in every chart)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [row["config"] for row in summary]
    width = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3.5, 4.2), dpi=150)
    for position, row in enumerate(summary):
        color = SERIES_COLORS[CONFIGS.index(row["config"])]
        xs = [index + (position - (len(configs) - 1) / 2) * width for index in range(len(metrics))]
        values = [0.0 if math.isnan(row[key]) else row[key] for key in metrics]
        bars = ax.bar(xs, values, width * 0.92, color=color, label=CONFIG_LABELS[row["config"]])
        for bar, key in zip(bars, metrics):
            label = "n/a" if math.isnan(row[key]) else (f"{row[key]:.2f}" if "mrr" in key else f"{row[key]:.0%}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, label,
                    ha="center", va="bottom", fontsize=7.5, color="#3d3d3a")
    ax.set_xticks(range(len(metrics)), list(metrics.values()))
    ax.set_ylim(0, 1.08)
    ax.set_title(title, loc="left", fontsize=11, color="#141413")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor="#6b6a64")
    ax.tick_params(axis="x", labelcolor="#3d3d3a")
    ax.legend(frameon=False, fontsize=8, ncol=len(configs), loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_components(configs: list[str], need_answers: bool, index_dir: Path) -> tuple[Any, Any, Any]:
    """Load only the models the chosen setups need. Returns (dense_index, reranker, generator).

    The generator from answer_generation.py already builds a HybridRetrieveRerank, and
    that already loads the DenseIndex, so each model is loaded at most once.
    """
    dense_index = reranker = generator = None
    needs_dense = any(config != "keyword" for config in configs)
    if need_answers:
        from answer_generation import QwenGeneratorRetail

        generator = QwenGeneratorRetail()
        reranker = generator.docRetriever
    elif "hybrid_rerank" in configs:
        from hybrid_retrieve_rerank import HybridRetrieveRerank

        reranker = HybridRetrieveRerank(index_dir=index_dir)
    if reranker is not None:
        dense_index = reranker.dense_index
    elif needs_dense:
        from dense_index import DenseIndex

        dense_index = DenseIndex.load(index_dir)
    return dense_index, reranker, generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--skip-answers", action="store_true", help="Only compute retrieval metrics (no LLM)")
    parser.add_argument("--limit", type=int, help="Use only the first N dev questions (quick trial runs)")
    parser.add_argument("--manual-labels", type=Path, help="answers_to_review.csv with the correct column filled in")
    args = parser.parse_args()

    questions = load_dev_questions(args.questions, args.split)
    if args.limit:
        questions = questions[: args.limit]
    configs = [config for config in CONFIGS if config in args.configs]
    print(f"Dev split: {len(questions)} questions. Setups: {', '.join(configs)}\n")

    dense_index, reranker, generator = build_components(configs, not args.skip_answers, args.index_dir)
    retrievers = Retrievers(load_jsonl(args.corpus), dense_index=dense_index, reranker=reranker)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Retrieval metrics")
    retrieval_summary, retrieval_rows = evaluate_retrieval(retrievers, questions, configs)
    write_csv(args.output_dir / "retrieval_metrics.csv", retrieval_summary)
    write_jsonl(args.output_dir / "retrieval_per_question.jsonl", retrieval_rows)
    plot_metrics(
        retrieval_summary,
        {f"hit_at_{HIT_K}": f"Hit@{HIT_K}", f"mrr_at_{MRR_K}": f"MRR@{MRR_K}"},
        f"Retrieval on the dev split ({len(questions)} questions)",
        args.output_dir / "retrieval_metrics.png",
    )

    answer_summary: list[dict[str, Any]] = []
    if generator is not None:
        from transformers import set_seed

        print("\nAnswer metrics (Qwen generator from answer_generation.py)")
        answer_summary = evaluate_answers(
            generator, retrievers, questions, configs, args.output_dir,
            load_manual_labels(args.manual_labels), seed_fn=set_seed,
        )
        write_csv(args.output_dir / "answer_metrics.csv", answer_summary)
        plot_metrics(
            answer_summary,
            {
                "answer_accuracy": "Answer accuracy",
                "citation_correctness": "Citation correctness",
                "unsupported_answer_rate": "Unsupported answers\n(lower is better)",
            },
            f"Answers on the dev split ({len(questions)} questions)",
            args.output_dir / "answer_metrics.png",
        )

    summary = {
        "split": "dev",
        "questions": len(questions),
        "settings": {
            "hit_k": HIT_K,
            "mrr_k": MRR_K,
            "candidates_per_retriever": CANDIDATES_PER_RETRIEVER,
            "rrf_k": RRF_K,
            "answer_top_k": ANSWER_TOP_K,
            "seed": SEED,
        },
        "retrieval": retrieval_summary,
        "answers": answer_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(without_nan(summary), indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
