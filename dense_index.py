"""Build and query a dense (semantic) FAISS index over the FinanceBench corpus.

Each corpus page is split into overlapping word-window passages so that no text
is lost to the embedding model's input limit. Every passage keeps the page and
document metadata from ``corpus_pages.jsonl`` (page_id, doc_name, company,
doc_type, doc_period, gics_sector, page_num, doc_link).

Build the index:
    python3 dense_index.py build

Query it:
    python3 dense_index.py search "What was 3M's FY2018 capital expenditure?"

Outputs (in ``indexes/dense/`` by default):
    faiss.index      FAISS inner-product index of L2-normalized embeddings
                     (inner product == cosine similarity)
    chunks.jsonl     one row per vector, in index order, with text + metadata
    config.json      model name, chunking settings, and counts used to build it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.joinpath("data", "processed", "corpus_pages.jsonl")
DEFAULT_INDEX_DIR = ROOT.joinpath("indexes", "dense")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_WORDS = 150
DEFAULT_OVERLAP_WORDS = 30

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
CONFIG_FILE = "config.json"

METADATA_FIELDS = (
    "page_id",
    "doc_name",
    "company",
    "doc_type",
    "doc_period",
    "gics_sector",
    "page_num",
    "doc_link",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split text into windows of ``chunk_words`` words that overlap by ``overlap_words``."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return windows


def make_chunks(
    pages: Iterable[dict[str, Any]],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Turn corpus pages into passage chunks, copying page/document metadata onto each."""
    chunks = []
    seen_page_ids = set()
    for page in pages:
        page_id = page["page_id"]
        if page_id in seen_page_ids:
            raise ValueError(f"Duplicate page_id in corpus: {page_id}")
        seen_page_ids.add(page_id)

        metadata = {field: page.get(field) for field in METADATA_FIELDS}
        for chunk_index, text in enumerate(split_words(page["text"], chunk_words, overlap_words)):
            chunks.append(
                {
                    "chunk_id": f"{page_id}_c{chunk_index}",
                    "chunk_index": chunk_index,
                    **metadata,
                    "text": text,
                }
            )
    return chunks


def embed(model: Any, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts as float32, L2-normalized vectors (so inner product = cosine)."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
    )
    return np.ascontiguousarray(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> Any:
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save(index: Any, chunks: list[dict[str, Any]], config: dict[str, Any], index_dir: Path) -> None:
    import faiss

    if index.ntotal != len(chunks):
        raise ValueError("Index size and chunk count do not match")
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (index_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


class DenseIndex:
    """A loaded FAISS index plus its chunk metadata and embedding model."""

    def __init__(self, index: Any, chunks: list[dict[str, Any]], model: Any) -> None:
        if index.ntotal != len(chunks):
            raise ValueError("Index size and chunk count do not match")
        self.index = index
        self.chunks = chunks
        self.model = model

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR, model: Any = None) -> "DenseIndex":
        import faiss

        config = json.loads((index_dir / CONFIG_FILE).read_text(encoding="utf-8"))
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config["model_name"])
        return cls(
            faiss.read_index(str(index_dir / INDEX_FILE)),
            load_jsonl(index_dir / CHUNKS_FILE),
            model,
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k chunks (with metadata and cosine score) for a query."""
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(embed(self.model, [query]), k)
        return [
            {**self.chunks[i], "score": float(score)}
            for score, i in zip(scores[0], indices[0])
            if i != -1
        ]

    def search_pages(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k distinct pages, each scored by its best-matching chunk."""
        hits = self.search(query, k=len(self.chunks))
        pages: dict[str, dict[str, Any]] = {}
        for hit in hits:  # already sorted best-first
            if hit["page_id"] not in pages:
                pages[hit["page_id"]] = hit
                if len(pages) == k:
                    break
        return list(pages.values())


def build(
    corpus_path: Path,
    index_dir: Path,
    model_name: str = DEFAULT_MODEL,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    model: Any = None,
) -> dict[str, Any]:
    pages = load_jsonl(corpus_path)
    chunks = make_chunks(pages, chunk_words, overlap_words)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    vectors = embed(model, [chunk["text"] for chunk in chunks])
    index = build_index(vectors)
    config = {
        "model_name": model_name,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "similarity": "cosine (inner product of L2-normalized vectors)",
        "faiss_index_type": "IndexFlatIP",
        "embedding_dim": int(vectors.shape[1]),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "corpus": str(corpus_path.relative_to(ROOT)) if corpus_path.is_relative_to(ROOT) else str(corpus_path),
    }
    save(index, chunks, config, index_dir)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="Embed the corpus and write the FAISS index")
    build_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    build_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    build_parser.add_argument("--model", default=DEFAULT_MODEL)
    build_parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    build_parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)

    search_parser = sub.add_parser("search", help="Query an existing index")
    search_parser.add_argument("query")
    search_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    search_parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "build":
        config = build(args.corpus, args.index_dir, args.model, args.chunk_words, args.overlap_words)
        print(f"Indexed {config['chunk_count']} chunks from {config['page_count']} pages "
              f"({config['embedding_dim']}-dim, {config['model_name']})")
        print(f"Saved to {args.index_dir}")
    else:
        dense = DenseIndex.load(args.index_dir)
        for rank, hit in enumerate(dense.search(args.query, args.k), start=1):
            preview = hit["text"][:160].replace("\n", " ")
            print(f"{rank}. [{hit['score']:.3f}] {hit['doc_name']} p.{hit['page_num']} ({hit['chunk_id']})")
            print(f"   {preview}...")


if __name__ == "__main__":
    main()
