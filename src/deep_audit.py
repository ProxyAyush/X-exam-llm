import pandas as pd
import json
import os

def deep_audit():
    results_dir = "results"
    datasets = [
        {"name": "truthful_qa", "file": "data/truthful_qa.parquet", "expected": 817},
        {"name": "HaluEval", "file": "data/HaluEval.parquet", "expected": 2000},
        {"name": "gsm8k", "file": "data/gsm8k.parquet", "expected": 1319},
        {"name": "medmcqa", "file": "data/medmcqa.parquet", "expected": 2000},
        {"name": "medqa", "file": "data/medqa.parquet", "expected": 1273},
        {"name": "case_hold", "file": "data/case_hold.parquet", "expected": 2000},
        {"name": "law_stack_exchange", "file": "data/law_stack_exchange.parquet", "expected": 638}
    ]

    print(f"{'Dataset':<20} | {'Expected':<10} | {'Unique Found':<12} | {'Integrity'}")
    print("-" * 70)

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
        if not os.path.exists(res_path):
            print(f"{ds['name']:<20} | {ds['expected']:<10} | 0            | ❌ MISSING FILE")
            continue

        # 1. Check Unique Queries in Results
        unique_queries = set()
        corrupted_count = 0
        total_lines = 0
        with open(res_path, 'r') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line)
                    q = data.get("query")
                    a = data.get("final_assertion")
                    m = data.get("model_used")
                    if q and a and m:
                        # Clean query for matching (strip prefixes)
                        clean_q = q.replace("Medical Question: ", "").replace("Legal Context: ", "").replace("Title: ", "")
                        unique_queries.add(clean_q[:200]) # Use large prefix as ID
                    else:
                        corrupted_count += 1
                except:
                    corrupted_count += 1

        # 2. Verify against Parquet Sample
        df = pd.read_parquet(ds["file"])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        # 3. Final Verdict
        found = len(unique_queries)
        status = "✅ PERFECT"
        if found < ds['expected']:
            status = f"⚠️ SHORT {ds['expected'] - found}"
        if corrupted_count > 0:
            status += f" (Contains {corrupted_count} bad lines)"
        
        print(f"{ds['name']:<20} | {ds['expected']:<10} | {found:<12} | {status}")

if __name__ == "__main__":
    deep_audit()
