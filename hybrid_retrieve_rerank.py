from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder

from dense_index import DenseIndex, DEFAULT_INDEX_DIR, load_jsonl

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.joinpath("data", "processed", "corpus_pages.jsonl")

"""
    Purpose of this py file: Combine TF-IDF keyword search and FAISS dense search through hybrid retrieval.
    Then use Cross-Encoder re-ranking.
"""
class HybridRetrieveRerank:
    def __init__(self, corpus_path: Path = DEFAULT_CORPUS, index_dir: Path = DEFAULT_INDEX_DIR):
        self.corpus = load_jsonl(corpus_path)
        self.page_ids = [row["page_id"] for row in self.corpus]

        # 

