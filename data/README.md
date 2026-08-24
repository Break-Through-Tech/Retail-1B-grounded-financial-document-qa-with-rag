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
