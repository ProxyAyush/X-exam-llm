import pandas as pd
import json
import os

def audit_phase3():
    results_dir = "results"
    datasets = [
        {"name": "truthful_qa", "file": "data/truthful_qa.parquet", "q_col": "question"},
        {"name": "HaluEval", "file": "data/HaluEval.parquet", "q_col": "question"},
        {"name": "gsm8k", "file": "data/gsm8k.parquet", "q_col": "question"},
        {"name": "medmcqa", "file": "data/medmcqa.parquet", "q_col": "question"},
        {"name": "medqa", "file": "data/medqa.parquet", "q_col": "question"},
        {"name": "case_hold", "file": "data/case_hold.parquet", "q_col": "context"},
        {"name": "law_stack_exchange", "file": "data/law_stack_exchange.parquet", "q_col": "title"}
    ]

    print(f"{'Dataset':<20} | {'Found':<10} | {'Status'}")
    print("-" * 50)

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
        if not os.path.exists(res_path):
            print(f"{ds['name']:<20} | 0          | ❌ MISSING FILE")
            continue

        # Load original data sample
        df = pd.read_parquet(ds["file"])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        # Load all results as a single massive string for greedy search
        with open(res_path, 'r') as f:
            all_results_text = f.read()

        missing_indices = []
        for i, row in df.iterrows():
            # Get the core question text
            if ds['name'] == "medqa":
                # MedQA is nested, extract the 'Question' field
                data = row.get('data', {})
                if isinstance(data, str):
                    try: data = json.loads(data)
                    except: data = {}
                core_text = data.get('Question', '')
            else:
                core_text = str(row.get(ds['q_col'], ''))

            # Check if this core text exists ANYWHERE in the results file
            # Use a substantial snippet to avoid false positives
            snippet = core_text[:100]
            if snippet not in all_results_text:
                missing_indices.append(i)

        found_count = len(df) - len(missing_indices)
        status = "✅ PERFECT" if len(missing_indices) == 0 else f"⚠️ MISSING {len(missing_indices)}"
        print(f"{ds['name']:<20} | {found_count:<10} | {status}")

if __name__ == "__main__":
    audit_phase3()
