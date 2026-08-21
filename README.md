# Extracting Insight from Financial Documents Using RAG

This repository supports the Break Through Tech AI Studio challenge project for building a grounded financial document question-answering system with retrieval-augmented generation (RAG).

The project focuses on answering questions from public financial filings while returning evidence-backed citations. The goal is not just to generate fluent answers, but to make the system traceable, reproducible, and careful about saying when the available documents do not support an answer.

## Project Team

| Name | Role | Contribution |
| :--- | :--- | :--- |
| SaiSandeep Kantareddy | Challenge Advisor | Project scope, evaluation guidance, milestone planning |
| 7-Eleven AI Studio Fellows | Project Team | Implementation, experiments, analysis, final documentation |

The fellow roster and individual contributions should be updated as the team begins implementation.

## Project Highlights

- Builds a RAG pipeline for grounded question answering over public financial documents.
- Uses the public FinanceBench sample as the initial benchmark dataset.
- Compares keyword retrieval, dense retrieval, reranking, answer generation, and citation quality.
- Evaluates the system with retrieval, answer correctness, citation support, abstention, latency, and reproducibility metrics.
- Keeps the project educational and public-data only; no proprietary 7-Eleven data or internal systems are used.

## Repository Contents

| Path | Purpose |
| :--- | :--- |
| [`Challenge-Project-Overview.md`](Challenge-Project-Overview.md) | Full challenge scope, deliverables, milestones, success criteria, and resources |
| [`Getting-Started-for-Fellows.md`](Getting-Started-for-Fellows.md) | Program guidance for fellows using this repository |
| [`data/README.md`](data/README.md) | Dataset provenance, validation, and loading notes |
| [`data/financebench_merged.jsonl`](data/financebench_merged.jsonl) | Frozen public FinanceBench sample used to start the project |
| [`requirements.txt`](requirements.txt) | Starter Python dependencies for data processing, retrieval, and evaluation |

## Setup and Installation

Clone the repository:

```bash
git clone https://github.com/Break-Through-Tech/7-Eleven-1B-grounded-financial-document-qa-with-rag.git
cd 7-Eleven-1B-grounded-financial-document-qa-with-rag
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate that the included dataset is readable:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/financebench_merged.jsonl")
rows = [json.loads(line) for line in path.open()]
print(f"Loaded {len(rows)} examples")
print(rows[0]["question"])
PY
```

## Project Overview

Financial filings are long, dense, and difficult to search manually. This project asks the team to build a system that can retrieve relevant evidence from public filings and generate concise answers with citations.

The first version should handle single-document questions over text and text extracted from tables. Multi-document reasoning, advanced table extraction, user interfaces, and model comparisons are stretch goals.

See [`Challenge-Project-Overview.md`](Challenge-Project-Overview.md) for the complete challenge description.

## Dataset

The starter dataset is the open-source public sample of [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench), included at [`data/financebench_merged.jsonl`](data/financebench_merged.jsonl).

Key details:

- 150 public examples
- JSON Lines format
- Includes questions, reference answers, evidence passages, source document names, page numbers, and document links
- Intended for benchmarking retrieval, grounded answering, and citation quality

See [`data/README.md`](data/README.md) for provenance, checksum, schema notes, and loading instructions.

## Model Development Plan

The expected workflow is:

1. Profile FinanceBench and create a stable development/test split.
2. Build a TF-IDF or BM25 retrieval baseline.
3. Add dense retrieval with Sentence Transformers and FAISS or cosine similarity.
4. Add reranking for the top retrieved passages.
5. Generate answers from retrieved context with page or passage citations.
6. Add abstention behavior when evidence is missing or weak.
7. Evaluate retrieval, answer correctness, citation support, and latency.

## Evaluation Plan

The final system should report results on a frozen test set. Target metrics are documented in the challenge overview and include:

- Hit@1, Hit@5, MRR@10 or nDCG@10 for retrieval
- Answer accuracy using normalized exact or numeric match where appropriate
- Human-rubric accuracy for qualitative answers
- Citation correctness and completeness
- Unsupported-answer rate
- Median and p95 latency

## Results and Key Findings

Results will be added after the team implements the baseline, experiments with retrieval and generation approaches, and evaluates the final system on the frozen test split.

## Next Steps

- Confirm the development/test split and save test IDs.
- Implement the first keyword retrieval baseline.
- Add retrieval evaluation scripts.
- Compare dense retrieval and reranking approaches.
- Save experiment outputs and document error categories.
- Prepare the final demo notebook or lightweight application.

## License

License terms should be confirmed with the Challenge Advisor and program staff before final publication.

## References

- [FinanceBench dataset card](https://huggingface.co/datasets/PatronusAI/financebench)
- [FinanceBench reference repository](https://github.com/patronus-ai/financebench)
- [FinanceBench paper](https://arxiv.org/abs/2311.11944)
- [SEC EDGAR filing search](https://www.sec.gov/search-filings)
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [FAISS getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

## Acknowledgements

This project is part of the Break Through Tech AI Studio program in collaboration with 7-Eleven.
