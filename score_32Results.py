"""
tuning_rag.py helped to see what min score should be.
run_frozen_test.py is what runs the 32 frozen test set questions.

This file is what checks criteria:

Success Criteria
Establish the exact train/development/test split and baseline during September, before tuning. Keep at least 30 of the 150 FinanceBench examples as a frozen test set. Where possible, split by source document so that questions about the same filing do not appear in both development and test sets.

Report all results on the same frozen test set. A successful final system should:

Achieve Hit@5 of at least 70% for retrieving a gold-evidence passage and improve Hit@5 by at least 10 percentage points over the keyword baseline.
Achieve answer accuracy of at least 55%, using normalized exact or numeric match where applicable and a documented human rubric for qualitative answers.
Achieve citation correctness of at least 85%, measured as the percentage of cited passages that support the associated answer.
Keep the unsupported-answer rate at or below 10%. When evidence is missing or below a tuned confidence threshold, the system should state that it cannot answer from the available documents.
Be reproducible from a fresh environment using documented commands, fixed random seeds, version-constrained dependencies, and saved evaluation outputs.
These thresholds are initial targets, not guarantees. If a target proves infeasible, the final report must explain why with quantitative evidence and error analysis. In addition to the headline criteria, report MRR@10 or nDCG@10, answer accuracy by question type, latency, and the number of model/API calls per question.

"""

import json
import re
from pathlib import Path

RESULTS_PATH = Path("results/frozen_test_results.json")

def normalize_number(text: str) -> str:
    # Extract digits/decimals for financial number comparison
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return "".join(nums).replace(",", "")

def main():
    if not RESULTS_PATH.exists():
        print("Run your frozen test script first to generate frozen_test_results.json!")
        return

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results = data["results"]
    total = len(results)

    correct_answers = 0
    correct_citations = 0
    valid_non_abstained = 0

    print("\n=== EVALUATING FROZEN TEST RESULTS ===")
    for item in results:
        qid = item["id"]
        gold_ans = str(item["gold_answer"]).strip()
        pred_ans = str(item["extracted_final_answer"]).strip()
        gold_pages = set(item["gold_evidence_pages"])
        cited_pages = set(item["cited_pages"])
        abstained = item["abstained"]

        #   Check Accuracy
        gold_norm = normalize_number(gold_ans)
        pred_norm = normalize_number(pred_ans)
        
        is_correct = False
        if gold_norm and pred_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            is_correct = True
        elif gold_ans.lower() in pred_ans.lower():
            is_correct = True

        if is_correct and not abstained:
            correct_answers += 1

        #  Check Citation 
        if not abstained and cited_pages:
            valid_non_appstained_count = 1
            valid_non_abstained += 1
            # If at least one cited page is in gold pages
            if cited_pages & gold_pages:
                correct_citations += 1

    accuracy_pct = (correct_answers / total) * 100
    citation_pct = (correct_citations / valid_non_abstained * 100) if valid_non_abstained > 0 else 0

    print(f"Total Questions:       {total}")
    print(f"Hit@5 Retrieval:       {data['hit_at_5_pct']}%")
    print(f"Estimated Accuracy:    {accuracy_pct:.1f}% ({correct_answers}/{total}) [Target >= 55%]")
    print(f"Citation Correctness:  {citation_pct:.1f}% ({correct_citations}/{valid_non_abstained}) [Target >= 85%]")
    print(f"Abstention Rate:       {data['abstention_rate_pct']}% [Target <= 10% unsupported]")
    print("======================================\n")

if __name__ == "__main__":
    main()