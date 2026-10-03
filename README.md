# Extracting Insight from Financial Documents Using RAG

Break Through Tech AI Studio, Fall 2026 | Retail 1B

## About this project

We are at the beginning of this project.

Our goal is to explore how retrieval-augmented generation, or RAG, could help people search public financial documents and find evidence for an answer. We will use FinanceBench to understand the type of questions and financial information the project will involve.

We have not built the RAG system yet. Right now, we are focused on understanding the project, the data, and what our first simple baseline should look like.

## Current milestone

Our September milestone is to define the project scope, review the FinanceBench data, and create a starting baseline before we begin building RAG.

| Task | Target date |
| --- | --- |
| Define the project scope and what the first version will cover | September 15 |
| Review the FinanceBench data | September 18 |
| Prepare the FinanceBench data | September 22 |
| Build a simple keyword-search baseline | September 26 |
| Test the baseline and review the results | September 29 |

Task ownership and day-to-day progress are tracked in our team milestone document and GitHub Issues.

## Keyword baseline

Run `python3 keyword_baseline.py` after the prepared data files are present. It evaluates the frozen test split and writes a JSON report to `results/keyword_baseline.json`. The current frozen run is Hit@5 62.5% (20 of 32 questions).

## Data

This repository includes a public FinanceBench sample:

[`data/financebench_merged.jsonl`](data/financebench_merged.jsonl)

The file contains 150 examples with financial questions, answers, evidence passages, and source-document information.

For more information about the dataset, see [`data/README.md`](data/README.md).

## Repository status

This repository currently contains:

```text
.
├── Challenge-Project-Overview.md
├── Getting-Started-for-Fellows.md
├── README.md
├── requirements.txt
└── data/
    ├── README.md
    └── financebench_merged.jsonl
```

As we start the project, we will add our notes, code, experiments, and results here.

## Resources

- [FinanceBench dataset](https://huggingface.co/datasets/PatronusAI/financebench)
- [FinanceBench reference repository](https://github.com/patronus-ai/financebench)
- [Project overview](Challenge-Project-Overview.md)
