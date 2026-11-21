"""Task #3 Milestone November: error analysis and test on frozen test set of 32 questions

"""
import json
import re
import time
from pathlib import Path

from answer_generation import QwenGeneratorRetail, INSUFFICIENT_EVIDENCE_RESPONSE
from dense_index import DenseIndex, DEFAULT_INDEX_DIR, load_jsonl

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "data" / "processed" / "eval_questions.jsonl"
SPLIT_PATH = ROOT / "data" / "processed" / "test_ids.json"
OUTPUT_PATH = ROOT / "results" / "frozen_test_results.json"

CITATION_PATTERN = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*Page\s+(\d+)\s*\]", re.I)
FINAL_ANSWER_PATTERN = re.compile(r"final answer\W*?[:\-]\s*(.+)", re.I)


def load_test_questions() -> list[dict]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_ids = set(split["test_ids"])
    by_id = {row["id"]: row for row in load_jsonl(QUESTIONS_PATH)}
    return [by_id[qid] for qid in sorted(test_ids)]


def extract_page_id(doc_name: str, page_num: str | int) -> str:
    return f"{doc_name.strip()}_p{str(page_num).strip()}"


def main():
    test_questions = load_test_questions()
    print(f"Loaded {len(test_questions)} frozen test questions.")

    generator = QwenGeneratorRetail(min_evidence_score=0.28)
    dense = generator.docRetriever

    results = []
    hit_at_5_count = 0
    abstain_count = 0
    latencies = []

    for i, item in enumerate(test_questions, start=1):
        qid = item["id"]
        query = item["question"]
        gold_answer = item["answer"]
        gold_pages = set(item.get("gold_evidence_pages", []))
        q_type = item.get("question_type", "unknown")

        start_time = time.perf_counter()

        # Check top-5 retrieval for Hit@5
        top_chunks = dense.search(query, k=5)
        retrieved_pages = {
            extract_page_id(c["doc_name"], c["page_num"]) for c in top_chunks
        }
        is_hit = bool(gold_pages & retrieved_pages)
        if is_hit:
            hit_at_5_count += 1

        # Run full answer generation
        raw_response = generator.generate(query, top_k=5)
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

        abstained = INSUFFICIENT_EVIDENCE_RESPONSE.lower() in raw_response.lower()
        if abstained:
            abstain_count += 1

        # Extract final answer line & citations
        final_match = FINAL_ANSWER_PATTERN.search(raw_response)
        final_line = final_match.group(1).strip() if final_match else ""
        citations = [
            extract_page_id(doc, page)
            for doc, page in CITATION_PATTERN.findall(raw_response)
        ]

        print(
            f"[{i}/{len(test_questions)}] {qid} | Hit@5: {is_hit} | "
            f"Abstained: {abstained} | {elapsed:.2f}s"
        )

        results.append({
            "id": qid,
            "question_type": q_type,
            "question": query,
            "gold_answer": gold_answer,
            "gold_evidence_pages": sorted(gold_pages),
            "retrieved_pages_top5": sorted(retrieved_pages),
            "top_score": round(top_chunks[0]["score"], 4) if top_chunks else None,
            "hit_at_5": is_hit,
            "abstained": abstained,
            "extracted_final_answer": final_line,
            "cited_pages": citations,
            "full_response": raw_response,
            "latency_sec": round(elapsed, 3),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_questions": len(test_questions),
        "hit_at_5_pct": round((hit_at_5_count / len(test_questions)) * 100, 2),
        "abstention_rate_pct": round((abstain_count / len(test_questions)) * 100, 2),
        "mean_latency_sec": round(sum(latencies) / len(latencies), 3),
        "model_calls_per_question": 2,  # 1 dense query embed + 1 Qwen generation
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FROZEN TEST RUN COMPLETE ===")
    print(f"Hit@5:           {summary['hit_at_5_pct']}% ({hit_at_5_count}/{len(test_questions)})")
    print(f"Abstention Rate: {summary['abstention_rate_pct']}% ({abstain_count}/{len(test_questions)})")
    print(f"Mean Latency:    {summary['mean_latency_sec']}s")
    print(f"Saved full output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()