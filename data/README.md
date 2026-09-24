# FinanceBench Data

This directory contains the public FinanceBench sample used for the Challenge Project.

## Included file

`financebench_merged.jsonl`

- Source: <https://huggingface.co/datasets/PatronusAI/financebench>
- Upstream filename: `financebench_merged.jsonl`
- Downloaded: 2026-08-20
- Rows: 150
- Size: 958,087 bytes
- SHA-256: `7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b`

Each line is a JSON object containing the benchmark question, gold answer, human justification, evidence passage(s), company and filing metadata, and a source-document URL. This is the 150-example open-source sample; it is not the full FinanceBench corpus.

## Load with pandas

```python
import pandas as pd

financebench = pd.read_json("data/financebench_merged.jsonl", lines=True)
print(financebench.shape)
print(financebench.columns.tolist())
```

## Verify the download

On macOS:

```bash
shasum -a 256 data/financebench_merged.jsonl
```

On Linux:

```bash
sha256sum data/financebench_merged.jsonl
```

## Source documents

The benchmark rows include evidence text and `doc_link` values. The source PDFs are also available from the [FinanceBench reference repository](https://github.com/patronus-ai/financebench/tree/main/pdfs). Download only the documents needed for the agreed project scope and retain document name, page number, period, and source URL as chunk metadata.

Review and follow the upstream dataset terms and citation guidance before redistributing or publishing derived artifacts. Do not add proprietary documents, credentials, or personal information to this public repository.


Updates after preparing data:

## Processed data (data/processed/)

The raw `financebench_merged.jsonl` sample is processed by 
`prepare_financebench.ipynb` into a set of derived files used by 
the RAG pipeline:

- **Chunking**: Evidence pages are deduplicated across questions and 
  stored as one chunk per unique (doc_name, page_number) pair, 
  preserving company, doc_type, doc_period, gics_sector, page 
  number, and source doc_link as metadata (`corpus_pages.jsonl`).
- **Eval set**: Questions and gold answers are kept separate from 
  the corpus, each linked to its gold evidence page(s) for 
  retrieval scoring (`eval_questions.jsonl`).
- **Train/test split**: Documents (not individual questions) are 
  randomly assigned to a test or dev group using a fixed seed (42), 
  so no filing appears in both groups. The test group is expanded 
  until it covers at least 30 questions. This split is saved to 
  `test_ids.json` and is frozen — it should not be regenerated or 
  tuned against until final evaluation.
- **Manifest**: `docs_manifest.csv` lists each unique source filing 
  with its metadata and original PDF link, for future use if full 
  filings are added beyond the sampled evidence pages.

To regenerate these files, run `prepare_financebench.ipynb` from a 
fresh clone with `requirements.txt` installed. Re-running with the 
same seed reproduces the identical split.