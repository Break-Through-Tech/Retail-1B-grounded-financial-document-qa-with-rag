from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder

from dense_index import DenseIndex, DEFAULT_INDEX_DIR

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.joinpath("data", "processed", "corpus_pages.jsonl")

"""
    Purpose: Combine TF-IDF keyword search and FAISS dense search through hybrid retrieval.
    Then use Cross-Encoder re-ranking.
"""

