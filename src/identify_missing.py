import pandas as pd
import json
import os

def identify_missing():
    results_dir = "results"
    datasets = [
        {"name": "truthful_qa", "file": "data/truthful_qa.parquet", "key": "question"},
        {"name": "HaluEval", "file": "data/HaluEval.parquet", "key": "question"},
        {"name": "gsm8k", "file": "data/gsm8k.parquet", "key": "question"},
        {"name": "medmcqa", "file": "data/medmcqa.parquet", "key": "question"},
        {"name": "medqa", "file": "data/medqa.parquet", "key": "medqa_logic"},
        {"name": "case_hold", "file": "data/case_hold.parquet", "key": "context"},
        {"name": "law_stack_exchange", "file": "data/law_stack_exchange.parquet", "key": "title"}
    ]

    missing_map = {}

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
        if not os.path.exists(res_path):
            df_full = pd.read_parquet(ds["file"])
            missing_map[ds['name']] = list(range(min(len(df_full), 2000)))
            continue

        df = pd.read_parquet(ds["file"])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        processed_set = set()
        with open(res_path, 'r') as f:
            for line in f:
                try:
                    processed_set.add(json.loads(line).get('query', ''))
                except: continue

        missing_indices = []
        for i, row in df.iterrows():
            # Reconstruct query using exact controller logic
            if ds['name'] == "medmcqa":
                options = f"A) {row.get('opa')}\nB) {row.get('opb')}\nC) {row.get('opc')}\nD) {row.get('opd')}"
                q_text = f"{row.get('question')}\nOptions:\n{options}"
            elif ds['name'] == "medqa":
                data = row.get('data')
                if isinstance(data, str): data = json.loads(data)
                question = data.get('Question', 'N/A')
                options = data.get('Options', {})
                options_str = "\n".join([f"{k}) {v}" for k, v in options.items()])
                q_text = f"Medical Question: {question}\nOptions:\n{options_str}"
            elif ds['name'] == "case_hold":
                options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(row.get('endings', []))])
                q_text = f"Legal Context: {row.get('context')}\nWhich of the following is the correct legal holding for this case?\nOptions:\n{options}"
            elif ds['name'] == "law_stack_exchange":
                q_text = f"Title: {row.get('title')}\nQuestion: {row.get('body')}"
            else:
                q_text = str(row.get(ds['key']))

            # Robust check
            if q_text not in processed_set:
                # Substring fallback for minor formatting shifts
                snippet = q_text[:100]
                found = False
                for p_q in processed_set:
                    if snippet in p_q:
                        found = True
                        break
                if not found:
                    missing_indices.append(i)
        
        if missing_indices:
            missing_map[ds['name']] = missing_indices
            print(f"Found {len(missing_indices)} missing items in {ds['name']}")

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/missing_indices.json", "w") as f:
        json.dump(missing_map, f, indent=2)
    print("Identification complete.")

if __name__ == "__main__":
    identify_missing()
