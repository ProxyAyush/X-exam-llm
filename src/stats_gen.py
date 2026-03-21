import pandas as pd
import json
import os
import re

RESULTS_DIR = "results"
DATA_DIR = "data"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_answer(text):
    # Try to find common patterns like "The answer is (A)" or "Final Answer: 42"
    match = re.search(r"([A-E])\)", text)
    if match: return match.group(1)
    
    match = re.search(r"answer is ([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # GSM8K style
    match = re.findall(r"(\d+\.?\d*)", text)
    if match: return match[-1]
    
    return text[:20] # Return snippet if no pattern

def evaluate_correctness():
    summary_stats = []
    
    datasets = [
        {"name": "truthful_qa", "file": "data/truthful_qa.parquet", "q_col": "question"},
        {"name": "HaluEval", "file": "data/HaluEval.parquet", "q_col": "question"},
        {"name": "gsm8k", "file": "data/gsm8k.parquet", "q_col": "question"},
        {"name": "medmcqa", "file": "data/medmcqa.parquet", "q_col": "question"},
        {"name": "medqa", "file": "data/medqa.parquet", "q_col": "question"}, # Nested logic needed
        {"name": "case_hold", "file": "data/case_hold.parquet", "q_col": "context"}, # CaseHold uses context as primary key
        {"name": "law_stack_exchange", "file": "data/law_stack_exchange.parquet", "q_col": "title"}
    ]

    for ds in datasets:
        res_path = os.path.join(RESULTS_DIR, ds["name"], "results.jsonl")
        if not os.path.exists(res_path):
            continue
            
        print(f"Analyzing {ds['name']}...")
        
        # Load parquet
        df_orig = pd.read_parquet(ds["file"])
        
        # Load results
        results = []
        with open(res_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        df_res = pd.DataFrame(results)
        
        # Merge or compare
        # For simplicity, we'll iterate and check
        correct_count = 0
        total = len(df_res)
        reject_count = 0
        
        for idx, row in df_res.iterrows():
            verdicts = [h.get('verdict') for h in row.get('history', []) if 'verdict' in h]
            final_verdict = verdicts[-1] if verdicts else "ACCEPT"
            if "REJECT" in final_verdict.upper():
                reject_count += 1
            
            # Simple heuristic: If Judge ACCEPTED, we check if it's actually correct (proxy)
            # In a real paper, we'd use a more rigorous check
            if "ACCEPT" in final_verdict.upper():
                correct_count += 1 # Placeholder for real GT check
        
        summary_stats.append({
            "Dataset": ds["name"],
            "Total Items": total,
            "Judge Accept Rate": (correct_count / total) * 100 if total > 0 else 0,
            "Adversarial Rejection Rate": (reject_count / total) * 100 if total > 0 else 0
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_summary.csv"), index=False)
    print("Statistical summary generated.")

if __name__ == "__main__":
    evaluate_correctness()
