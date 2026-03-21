import pandas as pd
import json
import os

def final_precise_audit():
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

    print(f"{'Dataset':<20} | {'Expected':<10} | {'Found':<10} | {'Status'}")
    print("-" * 65)

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
        if not os.path.exists(res_path):
            print(f"{ds['name']:<20} | 0          | 0          | ❌ MISSING")
            continue

        # Load samples
        df = pd.read_parquet(ds["file"])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        expected_count = len(df)
        
        # Load all processed snippets from results file
        processed_snippets = set()
        with open(res_path, 'r') as f:
            for line in f:
                try:
                    query = json.loads(line).get('query', '')
                    # Store a unique 50-char snippet to handle formatting variations
                    processed_snippets.add(query[:100])
                except: continue

        found_count = 0
        for i, row in df.iterrows():
            if ds['name'] == "medqa":
                data = row.get('data', {})
                if isinstance(data, str):
                    try: data = json.loads(data)
                    except: data = {}
                core_text = data.get('Question', '')
            else:
                core_text = str(row.get(ds['q_col'], ''))

            if core_text[:100] in processed_snippets:
                found_count += 1

        status = "✅ PERFECT" if found_count == expected_count else f"⚠️ MISSING {expected_count - found_count}"
        print(f"{ds['name']:<20} | {expected_count:<10} | {found_count:<10} | {status}")

if __name__ == "__main__":
    final_precise_audit()
