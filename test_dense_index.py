import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dense_index import METADATA_FIELDS, DenseIndex, build, make_chunks, split_words


class FakeModel:
    """Bag-of-words stand-in for a SentenceTransformer, so tests need no download."""

    vocab = ["assets", "income", "revenue", "cash", "debt"]

    def encode(self, texts, normalize_embeddings=True, **_):
        vectors = np.array(
            [[text.lower().count(word) + 0.01 for word in self.vocab] for text in texts],
            dtype="float32",
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


PAGES = [
    {
        "page_id": "ACME_2020_10K_p3",
        "doc_name": "ACME_2020_10K",
        "company": "ACME",
        "doc_type": "10k",
        "doc_period": 2020,
        "gics_sector": "Industrials",
        "page_num": 3,
        "doc_link": "https://example.com/acme.pdf",
        "text": "Total assets were 100 dollars. Assets grew.",
    },
    {
        "page_id": "ACME_2020_10K_p9",
        "doc_name": "ACME_2020_10K",
        "company": "ACME",
        "doc_type": "10k",
        "doc_period": 2020,
        "gics_sector": "Industrials",
        "page_num": 9,
        "doc_link": "https://example.com/acme.pdf",
        "text": "Net income was 5 dollars. Revenue was 50 dollars.",
    },
]


class SplitWordsTest(unittest.TestCase):
    def test_windows_overlap_and_cover_every_word(self):
        words = [f"w{i}" for i in range(10)]
        windows = split_words(" ".join(words), chunk_words=4, overlap_words=1)
        self.assertEqual(windows[0].split(), words[0:4])
        self.assertEqual(windows[1].split(), words[3:7])
        self.assertEqual(windows[-1].split()[-1], "w9")

    def test_short_text_is_one_window(self):
        self.assertEqual(split_words("a b c", 10, 2), ["a b c"])

    def test_rejects_overlap_not_smaller_than_window(self):
        with self.assertRaises(ValueError):
            split_words("a b c", 3, 3)


class MakeChunksTest(unittest.TestCase):
    def test_every_chunk_keeps_page_and_document_metadata(self):
        chunks = make_chunks(PAGES, chunk_words=4, overlap_words=1)
        self.assertGreater(len(chunks), len(PAGES))
        for chunk in chunks:
            page = next(p for p in PAGES if p["page_id"] == chunk["page_id"])
            for field in METADATA_FIELDS:
                self.assertEqual(chunk[field], page[field])
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))

    def test_rejects_duplicate_page_ids(self):
        with self.assertRaises(ValueError):
            make_chunks([PAGES[0], PAGES[0]])


class BuildAndSearchTest(unittest.TestCase):
    def test_build_save_load_and_search_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(p) + "\n" for p in PAGES))
            index_dir = tmp / "index"

            config = build(corpus, index_dir, model_name="fake", chunk_words=50, model=FakeModel())
            self.assertEqual(config["chunk_count"], 2)

            dense = DenseIndex.load(index_dir, model=FakeModel())
            top = dense.search("What were total assets?", k=1)[0]
            self.assertEqual(top["page_id"], "ACME_2020_10K_p3")
            self.assertEqual(top["page_num"], 3)
            self.assertEqual(top["doc_name"], "ACME_2020_10K")

            pages = dense.search_pages("revenue and income", k=5)
            self.assertEqual([p["page_id"] for p in pages][0], "ACME_2020_10K_p9")
            self.assertEqual(len(pages), 2)


if __name__ == "__main__":
    unittest.main()
