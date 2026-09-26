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

The [Challenge Advisor's project overview](Challenge-Project-Overview.md) calls for a TF-IDF or BM25 keyword baseline in September. This script tries both methods on the same prepared report pages. Each method uses the words in a question to rank pages and saves the top five page IDs and full page records. The search finds sources; it does not write an answer. The steps and inputs are explained in [`keyword_baseline.py`](keyword_baseline.py), and [`test_keyword_baseline.py`](test_keyword_baseline.py) has a small made-up example.

Install dependencies with `python3 -m pip install -r requirements.txt`, then run `python3 keyword_baseline.py` after the prepared data files are present. It checks the fixed test questions with TF-IDF and BM25, then writes `results/keyword_baseline.json` and `results/bm25_baseline.json`.

**Hit@1** is the share of questions where a correct source page ranks first. **Hit@5** is the share where a correct page appears among the top five. **MRR@10** looks at the first correct page in the top ten: first place scores 1, second place scores 0.5, and no correct page scores 0. It averages those scores across questions. The report still saves full records for only the top five pages.

On the same 32 questions, TF-IDF scored 34.4% Hit@1 (11/32), 62.5% Hit@5 (20/32), and 0.462 MRR@10. BM25 scored 43.8% Hit@1 (14/32), 68.8% Hit@5 (22/32), and 0.518 MRR@10. No parameters were tuned on the test set.

These scores come from 168 sampled evidence pages. The corpus does not yet include every page of each filing, so the scores may change when full filings are added.

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
