# Extracting Insight from Financial Documents Using RAG

## 🎯 The Challenge

### Project Summary 
**Challenge Advisor:** SaiSandeep Kantareddy, saisandeep.kantareddy@gmail.com

**AI Coach:** Alexandra Ladyzhensky, alexandra.ladyzhensky@breakthroughtech.org

**Program:** Break Through Tech AI Studio - Fall 2026

Financial filings are long, dense, and difficult to search manually. In this project, the team will build a retrieval-augmented generation (RAG) system that answers questions about public financial documents and supports each answer with traceable evidence from the source.

The team will ingest and chunk documents, compare keyword and embedding-based retrieval, add a reranking step, generate answers with citations, and evaluate the complete system. The result should be a reproducible reference implementation that helps a user find reliable facts more quickly while reducing unsupported or hallucinated answers.

The system should handle questions such as:

- **Direct extraction:** What was a company's capital expenditure in a specified fiscal year?
- **Comparison:** How did a metric such as revenue, debt, or operating margin change year over year?
- **Calculation:** What percentage of revenue was spent on research and development?
- **Evidence synthesis:** Based on the filing, what factors explain a reported change?
- **Insufficient evidence:** Can the system decline to answer when the indexed documents do not support a response?

The required scope is single-document questions over text and text extracted from tables. Multi-document reasoning and advanced table extraction are stretch goals.

### Deliverables

By the end of the challenge, the repository should include:

1. A documented data-ingestion and preprocessing workflow.
2. A keyword-search baseline and at least one dense or hybrid retrieval approach.
3. A RAG pipeline that returns an answer, document identifier, and page or passage citation.
4. A repeatable evaluation harness with a frozen test split and saved results.
5. Error analysis covering retrieval misses, calculation errors, unsupported claims, and citation failures.
6. Setup instructions, architecture documentation, and a final demo notebook or lightweight application.

### Success Criteria

Establish the exact train/development/test split and baseline during September, before tuning. Keep at least 30 of the 150 FinanceBench examples as a frozen test set. Where possible, split by source document so that questions about the same filing do not appear in both development and test sets.

Report all results on the same frozen test set. A successful final system should:

- Achieve **Hit@5 of at least 70%** for retrieving a gold-evidence passage and improve Hit@5 by **at least 10 percentage points** over the keyword baseline.
- Achieve **answer accuracy of at least 55%**, using normalized exact or numeric match where applicable and a documented human rubric for qualitative answers.
- Achieve **citation correctness of at least 85%**, measured as the percentage of cited passages that support the associated answer.
- Keep the **unsupported-answer rate at or below 10%**. When evidence is missing or below a tuned confidence threshold, the system should state that it cannot answer from the available documents.
- Be reproducible from a fresh environment using documented commands, fixed random seeds, version-constrained dependencies, and saved evaluation outputs.

These thresholds are initial targets, not guarantees. If a target proves infeasible, the final report must explain why with quantitative evidence and error analysis. In addition to the headline criteria, report MRR@10 or nDCG@10, answer accuracy by question type, latency, and the number of model/API calls per question.

### Stretch Goals

- Improve retrieval from tables while preserving row and column context.
- Support questions requiring evidence from multiple filings.
- Compare an open-source model with an API-based model under the same evaluation protocol.
- Calibrate an abstention threshold and plot the coverage-versus-accuracy trade-off.
- Add a small user interface for viewing answers, highlighted evidence, and pipeline traces.

### Project Milestones

Use these milestones to guide the work. The team will create a **GitHub Projects board** to track tasks and acceptance criteria within each milestone.

| Month | Milestone | Key Activities | Exit Criteria |
| :--- | :--- | :--- | :--- |
| September | **Scope, Data, and Baseline** | Profile FinanceBench; define the document and question scope; create a document-level development/test split; parse and chunk source material; implement a TF-IDF or BM25 baseline; define metric code and experiment logging. | Reproducible data pipeline, frozen test IDs, keyword baseline results, and documented risks. |
| October | **Grounded RAG Prototype** | Implement dense retrieval, a vector index, retrieve-and-rerank, answer generation, citation formatting, and insufficient-evidence behavior; compare configurations on the development set. | End-to-end pipeline returns answers with citations; automated retrieval and answer evaluation runs successfully. |
| November | **Evaluation, Hardening, and Demo** | Tune only on development data; evaluate once on the frozen test set; perform error and cost/latency analysis; improve documentation; prepare the final demo and presentation. | Final metrics and error analysis are saved; setup is reproducible; demo and project report are ready. |

---

## 📊 Dataset

- **Primary dataset:** [FinanceBench on Hugging Face](https://huggingface.co/datasets/PatronusAI/financebench)
- **Reference repository and source PDFs:** [Patronus AI FinanceBench](https://github.com/patronus-ai/financebench)
- **Optional supplemental source:** [SEC EDGAR filing search](https://www.sec.gov/search-filings)
- **Format:** JSON Lines containing questions, gold answers, evidence passages, metadata, and source-document links; source filings are generally PDF, HTML, or text
- **Size:** The included open-source benchmark file is approximately 958 KB and contains 150 examples; any curated document collection must remain under 1 GB
- **Repository location:** [`data/financebench_merged.jsonl`](data/financebench_merged.jsonl)

### Key Details

- Each row has a stable question ID plus company, filing, period, question, gold answer, justification, evidence passage(s), page number(s), and source URL.
- The questions include information extraction, numerical reasoning, and logical reasoning over 10-K, 10-Q, 8-K, and earnings materials.
- The included 150-example file is the complete **public sample**, not the full 10,231-question FinanceBench corpus described by the benchmark authors.
- The JSONL already contains gold evidence text, so the team can build an initial retrieval corpus without downloading every PDF. Source filings may be added selectively for realistic document parsing and page-level citations.
- Clean document text while preserving company, filing period, document type, source URL, page number, section, and table context as chunk metadata.
- Do not manually correct or tune against frozen test answers. Document any unavailable or changed source link and retain the benchmark evidence as the reproducible reference.
- If downloading additional filings programmatically, follow the [SEC's EDGAR API guidance](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), identify the client with an appropriate User-Agent, and respect SEC access policies.
- See [`data/README.md`](data/README.md) for provenance, validation, and loading instructions.

---

## 🛠️ Suggested Approach

**ML Problem Type:** Natural language processing, information retrieval, and large-language-model question answering

### Recommended Workflow

1. **Profile and split the data:** Inspect question and document types; create stable development and test IDs, preferably grouped by `doc_name`.
2. **Build the baseline:** Index chunks with `TfidfVectorizer` or BM25 and retrieve the top *k* passages.
3. **Add semantic retrieval:** Embed queries and chunks with a pretrained Sentence Transformers model; use cosine similarity or FAISS for top-*k* search.
4. **Rerank:** Score the retrieved candidates with a cross-encoder and preserve retrieval scores and metadata.
5. **Generate grounded answers:** Instruct the language model to answer only from retrieved context, show calculations, cite source/page, and abstain when evidence is insufficient.
6. **Evaluate by stage:** Evaluate retrieval independently before answer generation; then evaluate answer correctness, citation support, abstention, latency, and cost.
7. **Analyze errors:** Label failures by parsing, chunking, retrieval, reranking, reasoning/calculation, generation, and citation.

### Recommended Libraries

- `pandas` and `numpy` for dataset processing and analysis
- `scikit-learn` for the TF-IDF baseline and metric utilities
- `sentence-transformers` for embeddings and cross-encoder reranking
- `faiss-cpu` for local vector search (or a simple in-memory cosine-similarity index for the initial corpus)
- `pypdf` and `beautifulsoup4` for optional PDF/HTML parsing
- `pytest` for deterministic pipeline tests
- `jupyter` or Google Colab for exploration and demos
- An instructor-approved LLM SDK or a local Hugging Face model for answer generation; never commit API keys

### Evaluation Metrics

| Layer | Required Metrics | What They Measure |
| :--- | :--- | :--- |
| Retrieval | Hit@1, Hit@5, MRR@10 or nDCG@10 | Whether gold evidence is found and how highly it is ranked |
| Answer | Exact match / normalized numeric match; rubric-based accuracy for qualitative answers | Whether the final answer matches the reference, including units and direction |
| Citation | Citation correctness and citation completeness | Whether cited evidence supports the answer and whether major claims are cited |
| Safety | Unsupported-answer rate; answerable/unanswerable abstention precision and recall | Whether the system avoids inventing answers when evidence is weak |
| Operations | Median and p95 latency; model/API calls per question | Whether quality improvements have acceptable runtime and resource cost |

BLEU is not recommended as the primary answer metric because valid financial answers can be short, numeric, or phrased differently from the reference.

---

## 📚 Resources to Get Started

### Background Reading

- [FinanceBench paper: A New Benchmark for Financial Question Answering](https://arxiv.org/abs/2311.11944)
- [FinanceBench dataset card](https://huggingface.co/datasets/PatronusAI/financebench)
- [SEC EDGAR filing search](https://www.sec.gov/search-filings)

### Technical Tutorials and Documentation

- [Sentence Transformers: Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Sentence Transformers: Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Sentence Transformers: Information Retrieval Evaluation](https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html)
- [scikit-learn: Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [FAISS: Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

### Code Examples

- [FinanceBench reference repository](https://github.com/patronus-ai/financebench)
- [Sentence Transformers retrieve-and-rerank examples](https://github.com/huggingface/sentence-transformers/tree/master/examples/sentence_transformer/applications/retrieve_rerank)

Feel free to explore beyond these resources and share useful findings with the team.

---

## 🤝 How We'll Work Together

**Official check-ins:** During the biweekly 45-minute AI Studio Lab Section meeting block (second and fourth week of each month).

**Questions between check-ins:** Email the Challenge Advisor and copy all teammates and the AI Studio Coach so decisions remain visible to the full team. Group related questions, include links to the relevant issue or experiment, and allow up to two business days for a response. Use the team's Break Through Tech communication channel for routine coordination. For urgent program or access issues, contact the AI Studio Coach.

**Recommended free coding and collaboration tools:**

- GitHub Issues and GitHub Projects for tasks, decisions, and milestone tracking
- GitHub pull requests for reviewable code and documentation changes
- Google Colab for shared experiments that do not require local setup
- Google Drive or the program-approved shared workspace for non-code collaboration

Do not place API keys, credentials, proprietary information, or personal data in the public repository. Store secrets in environment variables or the notebook platform's secret manager.

---

## 🚀 Getting Started

1. Review this overview and [`Getting-Started-for-Fellows.md`](Getting-Started-for-Fellows.md); record questions for the first meeting.
2. Read [`data/README.md`](data/README.md), load the included JSONL file, and inspect the distribution of documents, question types, and evidence pages.
3. Create the GitHub Projects board and issues for the September exit criteria.
4. Agree on stable development and test IDs before tuning any retrieval or generation component.
5. Run a minimal baseline that retrieves evidence for a small set of questions and save the results.

I’m excited to work with you!

---

## ❓ Questions?

Please bring questions to our first meeting during the week of August 24, 2026 (Break Through Tech's Bridge to Studio — Session C), or use the communication process above.
