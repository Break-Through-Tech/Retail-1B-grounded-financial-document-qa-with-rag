import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hybrid_retrieve_rerank import HybridRetrieveRerank
import time
from pathlib import Path
from hybrid_retrieve_rerank import HybridRetrieveRerank
from dense_index import load_jsonl


# Load your files
data_file = Path("data/financebench_merged.jsonl")
my_dataset = load_jsonl(data_file)

# Initialize searcher
searcher = HybridRetrieveRerank()

# --- ADD THIS FILTER HERE ---
# Figure out which document names actually exist in your local index corpus
available_docs = {row["doc_name"] for row in searcher.corpus}

# Filter the dataset to only test questions for documents you've actually indexed
my_dataset = [row for row in my_dataset if row.get("doc_name") in available_docs]
print(f"Filtered dataset to {len(my_dataset)} questions that match your local indexed documents.\n")
# ----------------------------

hits = 0
num_questions = min(10, len(my_dataset))  # test up to 10 available questions
all_times = []

print(f"Testing retrieval on {num_questions} questions...\n")

for i, row in enumerate(my_dataset[:num_questions]):
    q = row["question"]
    target_page = row.get("evidence_page_num") 
    target_doc = row.get("doc_name")
    
    start_t = time.time()
    res = searcher.search(q, top_res=5)
    end_t = time.time()
    
    time_taken = (end_t - start_t) * 1000  
    all_times.append(time_taken)
    
    found_match = False
    for r in res:
        doc_matches = str(r.get("doc_name")).strip() == str(target_doc).strip()
        
        if target_page is None or target_page == "None" or target_page == "":
            if doc_matches:
                found_match = True
                break
        else:
            page_matches = str(r.get("page_num")).strip() == str(target_page).strip()
            if doc_matches and page_matches:
                found_match = True
                break
            
    if found_match:
        hits += 1

hit_at_5_score = (hits / num_questions) * 100 if num_questions > 0 else 0
average_time = sum(all_times) / len(all_times) if all_times else 0

print("--- Results ---")
print(f"Hit@5: {hit_at_5_score}%")
print(f"Avg Latency: {average_time:.2f} ms")