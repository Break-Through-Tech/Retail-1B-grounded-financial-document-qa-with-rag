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

        # Using TF-IDF keyword baseline

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.corpus_matrix = self.vectorizer.fit_transform([row["text"] for row in self.corpus])


        # Loading dense_index.py FAISS index for use

        self.dense_index = DenseIndex.load(index_dir)

        # Use Cross-Encoder here for rerank part
        self.rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    

    # Use tools gathered so far to get the top documents for task 3 prompting
    def search(self, query: str, top_res: int = 5) -> list[dict[str, Any]]:
        print("-> Running TF-IDF search...")
        tfidf_scores = (self.corpus_matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        tfidf_indices = tfidf_scores.argsort()[::-1][:20]
        keyword_hits = [self.corpus[i] for i in tfidf_indices]
        print("-> Running Dense FAISS search...")
        candidates_twenty = self.dense_index.search(query, k=20)
        # put values for top documents with page id and pair with query
        print("-> Combining and re-ranking with Cross-Encoder...")
        combined = {hit["page_id"]: hit for hit in keyword_hits + candidates_twenty}
        candidates = list(combined.values())
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.rerank.predict(pairs, show_progress_bar=False) # important because can evaluate if the top were good with metrics

        for c, score in zip(candidates, scores):
            c["hybrid_score"] = float(score)

        candidates.sort(key=lambda x:x["hybrid_score"], reverse=True)
        print("-> Re-ranking complete! Returning results...")
        return candidates[:top_res]



if __name__ == "__main__":
    import sys
    query = ''
    if (len(sys.argv) > 1):
        query = sys.argv[1]
    else:
        query = "What was 3M's FY2018 capital expenditure?"

    print(f"Running hybrid search for query: `{query}`\n")
    searcher = HybridRetrieveRerank()
    result = searcher.search(query, top_res=5)

    for rank, r in enumerate(result, start=1):
        print(f"{rank}. [Score: {r['hybrid_score']:.3f}] {r['doc_name']} (Page {r['page_num']})")
        print(f"   Snippet: {r['text'][:120]}...\n")
    print("Done! Task 2 search execution successful.")


