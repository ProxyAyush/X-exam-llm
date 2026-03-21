import pandas as pd
import json
import os

def identify_missing():
    results_dir = "results"
    # Target only the ones with gaps
    datasets = [
        {"name": "HaluEval", "file": "data/HaluEval.parquet", "q_col": "question"},
        {"name": "gsm8k", "file": "data/gsm8k.parquet", "q_col": "question"}
    ]

    missing_map = {}

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
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
            core_text = str(row.get(ds['q_col'], ''))
            if core_text[:100] not in processed_snippets:
                missing_indices.append(i)
        
        missing_map[ds['name']] = missing_indices
        print(f"Found {len(missing_indices)} missing items in {ds['name']}")

    with open("analysis/missing_indices.json", "w") as f:
        json.dump(missing_map, f, indent=2)

if __name__ == "__main__":
    identify_missing()
