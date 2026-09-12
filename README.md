# Extracting insight from financial documents using RAG

Break Through Tech AI Studio, Fall 2026 | Retail 1B

## What we are building

Financial filings are difficult to search by hand. This project will build a retrieval-augmented generation (RAG) system that answers questions about public financial documents and shows the evidence used for each answer.

The system will support single-document questions over filing text and text extracted from tables. It should handle direct facts, comparisons, calculations, evidence-based explanations, and cases where the available document does not support an answer.

Multi-document reasoning and advanced table extraction are stretch goals for later in the project.

## September plan: scope, data, and baseline

By September 30, the team will establish a reproducible starting point before building the full RAG pipeline.

| Task | Current owner(s) | Target date |
| --- | --- | --- |
| Define the first version's scope, test questions, and exclusions | Sean, Amy Rodriguez | September 15 |
| Review the FinanceBench data and create a stable development/test split | Sarayu | September 18 |
| Prepare FinanceBench documents as chunks with page and source metadata | Pranavi | September 22 |
| Build a keyword-search baseline that returns relevant passages | Sean | September 26 |
| Test the baseline on sample questions and review its weaknesses | TBD | September 29 |

The September deliverables are a reproducible data pipeline, frozen test IDs, keyword-baseline results, and documented risks. The team will track these tasks as GitHub Issues and add them to the coach-created GitHub Project board when it is available.

## Data

The repository includes the public FinanceBench sample at [`data/financebench_merged.jsonl`](data/financebench_merged.jsonl).

- 150 examples in JSON Lines format
- Questions, gold answers, human justifications, evidence passages, and filing metadata
- Company, filing, page, and source-document information for preserving evidence context

This file is the 150-example public sample, not the full FinanceBench benchmark. See [`data/README.md`](data/README.md) for provenance, loading instructions, and the file checksum.

## Planned approach

1. Profile FinanceBench and freeze a document-level development/test split.
2. Clean and chunk the available evidence while retaining source metadata.
3. Establish a TF-IDF or BM25 keyword-search baseline.
4. Compare that baseline with dense or hybrid retrieval.
5. Add reranking and grounded answer generation with citations.
6. Evaluate retrieval, answers, citations, abstention behavior, latency, and cost.

The final system should return an answer with a document identifier and page or passage citation. When retrieved evidence is weak or missing, it should say that it cannot answer from the available documents.

## Evaluation

Every approach will use the same frozen test set.

| Area | Measures |
| --- | --- |
| Retrieval | Hit@1, Hit@5, and MRR@10 or nDCG@10 |
| Answers | Normalized exact or numeric match, plus a documented rubric for qualitative answers |
| Citations | Whether citations support the answer and cover its main claims |
| Safety | Unsupported-answer rate and abstention performance |
| Operations | Latency and model or API calls per question |

The initial targets are at least 70% Hit@5, a 10-percentage-point improvement over the keyword baseline, at least 55% answer accuracy, and at least 85% citation correctness.

## Repository contents

```text
.
├── Challenge-Project-Overview.md   # Official project brief and requirements
├── Getting-Started-for-Fellows.md  # Break Through Tech setup guidance
├── README.md                       # Project overview
├── requirements.txt                # Python dependencies
└── data/
    ├── README.md                   # Dataset provenance and loading instructions
    └── financebench_merged.jsonl  # Public FinanceBench sample
```

The team will add implementation code, notebooks, experiment outputs, and evaluation results as the project develops.

## Setup

```bash
git clone https://github.com/Break-Through-Tech/Retail-1B-grounded-financial-document-qa-with-rag.git
cd Retail-1B-grounded-financial-document-qa-with-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To confirm that the included dataset loads:

```python
import pandas as pd

financebench = pd.read_json("data/financebench_merged.jsonl", lines=True)
print(financebench.shape)
print(financebench.columns.tolist())
```

Commands for preprocessing, retrieval, evaluation, and the final demo will be added with those components.

## Working together

- Use GitHub Issues to track work and GitHub pull requests to review changes.
- Keep experiment settings and outputs reproducible.
- Do not commit API keys, credentials, personal information, or proprietary documents.
- Record important decisions, data limitations, and evaluation findings in the repository.

## References

- [FinanceBench paper](https://arxiv.org/abs/2311.11944)
- [FinanceBench dataset](https://huggingface.co/datasets/PatronusAI/financebench)
- [FinanceBench reference repository](https://github.com/patronus-ai/financebench)
- [Sentence Transformers semantic search documentation](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [scikit-learn text analytics tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
