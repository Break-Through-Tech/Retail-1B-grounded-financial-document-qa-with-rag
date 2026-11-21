from pathlib import Path
from dense_index import DenseIndex, DEFAULT_INDEX_DIR
from compare_configurations import load_dev_questions, DEFAULT_QUESTIONS, DEFAULT_SPLIT

# Load 118 dev questions 
dev_questions = load_dev_questions(DEFAULT_QUESTIONS, DEFAULT_SPLIT)
print(f"Successfully loaded {len(dev_questions)} development questions.")

# Load the dense FAISS index
dense = DenseIndex.load(DEFAULT_INDEX_DIR)

# Check similarity scores
scores = []
for q in dev_questions:
    hits = dense.search(q["question"], k=1)
    scores.append(hits[0]["score"] if hits else -1.0)

scores.sort()

#Find the best threshold for <= 10% 
print(f"\nLowest 15 scores in the dev set: {[round(s, 3) for s in scores[:15]]}")
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    abstained = sum(1 for s in scores if s < th)
    rate = (abstained / len(scores)) * 100
    print(f"Threshold {th:.2f} -> Abstains on {abstained}/{len(scores)} questions ({rate:.1f}%)")