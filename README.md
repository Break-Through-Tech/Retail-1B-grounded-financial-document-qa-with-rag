# Extracting Insight from Financial Documents Using RAG

Break Through Tech AI Studio, Fall 2026 | Retail 1B

## Project Overview & Objectives

Financial filings (such as SEC 10-K and 10-Q documents) are notoriously long, dense, and difficult to search manually. The objective of this project was to build a robust Retrieval-Augmented Generation (RAG) system to answer complex financial questions over SEC filings while ensuring every claim is backed by traceable, page-level evidence and low hallucination rates.

The system addresses four core question types:
- **Direct extraction:** Locating specific financial metrics (e.g., capital expenditures).
- **Comparison:** Tracking year-over-year changes in revenue, debt, or operating margins.
- **Calculation:** Computing derived metrics (e.g., R&D percentage of revenue).
- **Evidence synthesis & Abstention:** Declining to answer when available documents lack sufficient supporting evidence.

---

## Methodology & Pipeline Architecture

1. **Data Preprocessing & Chunking (Task 1):** Cleaned SEC text filings into structured chunks with a rolling word window and overlap, retaining rich metadata (`company`, `doc_name`, `page_num`, `doc_period`) saved as a JSON Lines corpus (`data/financebench_merged.jsonl`).
2. **Retrieval & Reranking Evaluation (Task 2):** Compared a TF-IDF keyword baseline against dense vector embeddings (`SentenceTransformers`), hybrid search, and cross-encoder reranking. 
3. **Generation & Abstention (Task 3):** Integrated `Qwen2.5-3B-Instruct` with deterministic decoding (`do_sample=False`), device routing (`mps`/`cuda`/`cpu`), strict citation prompt formatting (`[DocName, Page X]`), and a tuned confidence threshold to handle insufficient evidence.

---

## Results & Key Findings

We evaluated our configurations on a development split of 118 questions and executed our final evaluation on a **frozen test split of 32 questions**. 

- **Hit@5 Retrieval:** Our dense retrieval setup achieved **77% Hit@5** on the development set, outperforming the TF-IDF keyword baseline (53%) by 24 percentage points[cite: 1].
- **Reranker Trade-offs:** While cross-encoder reranking provides deep semantic scoring, its strict threshold (`-3.0`) caused it to decline 59% of queries[cite: 1], leading us to select pure dense retrieval for our final architecture.
- **Citation & Accuracy Targets:** Through prompt engineering enforcing `[DocName, Page X]` citations and strict Yes/No formatting, we optimized citation correctness and controlled the unsupported-answer rate ($\le 10\%$).

---

## Visualizations & Error Analysis

During our November evaluation, we conducted a thorough error analysis on test failures, categorized into four primary buckets:
1. **Retrieval Misses:** Cases where gold evidence fell outside the top-$k$ retrieved passages.
2. **False Abstentions:** Cases where correct evidence was retrieved, but confidence scores fell below the abstention threshold.
3. **Math / Reasoning Errors:** Cases where the 3B model miscalculated financial figures or misread dense table structures.
4. **Citation Format Errors:** Omissions or slight mismatches in required `[DocName, Page X]` syntax.

*(Results, configuration comparison tables, and performance graphs are saved under `results/`)*.

---
## Repo Structure
.
├── Challenge-Project-Overview.md
├── Getting-Started-for-Fellows.md
├── README.md
├── requirements.txt
├── answer_generation.py
├── dense_index.py
├── hybrid_retrieve_rerank.py
├── data/
│   ├── README.md
│   └── financebench_merged.jsonl
└── results/
    └── evaluated_results.jsonl

## Local Setup & Installation

To run or reproduce this project:

```bash
# 1. Clone the repository and navigate inside
git clone [https://github.com/your-repo/retail-1b-rag.git](https://github.com/your-repo/retail-1b-rag.git)
cd retail-1b-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the dense vector index
python3 dense_index.py build

# 4. Run retrieval search or evaluate pipeline
python3 dense_index.py search "What was 3M's FY2018 capital expenditure?"
