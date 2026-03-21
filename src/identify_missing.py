import pandas as pd
import json
import os

def identify_missing():
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

    missing_map = {}

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
        
        # Load samples
        df = pd.read_parquet(ds["file"])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        processed_snippets = set()
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                for line in f:
                    try:
                        q = json.loads(line).get('query', '')
                        processed_snippets.add(q[:100])
                    except: continue

        missing_indices = []
        for i, row in df.iterrows():
            if ds['name'] == "medqa":
                data = row.get('data', {})
                if isinstance(data, str):
                    try: data = json.loads(data)
                    except: data = {}
                core_text = data.get('Question', '')
            else:
                core_text = str(row.get(ds['q_col'], ''))

            if core_text[:100] not in processed_snippets:
                missing_indices.append(i)
        
        if missing_indices:
            missing_map[ds['name']] = missing_indices
            print(f"Found {len(missing_indices)} missing items in {ds['name']}")

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/missing_indices.json", "w") as f:
        json.dump(missing_map, f, indent=2)

if __name__ == "__main__":
    identify_missing()
