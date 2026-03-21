import pandas as pd
import json
import os

def database_specific_audit():
    results_dir = "results"
    # Mapping unique logic for each
    datasets = [
        {"name": "truthful_qa", "file": "data/truthful_qa.parquet", "key": "question"},
        {"name": "HaluEval", "file": "data/HaluEval.parquet", "key": "question"},
        {"name": "gsm8k", "file": "data/gsm8k.parquet", "key": "question"},
        {"name": "medmcqa", "file": "data/medmcqa.parquet", "key": "question"},
        {"name": "medqa", "file": "data/medqa.parquet", "key": "medqa_logic"},
        {"name": "case_hold", "file": "data/case_hold.parquet", "key": "context"},
        {"name": "law_stack_exchange", "file": "data/law_stack_exchange.parquet", "key": "title"}
    ]

    print(f"{'Dataset':<20} | {'Found':<10} | {'Status'}")
    print("-" * 50)

    for ds in datasets:
        res_path = os.path.join(results_dir, ds["name"].replace("/", "_"), "results.jsonl")
        if not os.path.exists(res_path):
            print(f"{ds['name']:<20} | 0          | ❌ MISSING")
            continue

        df = pd.read_parquet(ds["file"])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        # Load all queries into a set for fast lookup
        processed_set = set()
        with open(res_path, 'r') as f:
            for line in f:
                try:
                    q = json.loads(line).get('query', '')
                    processed_set.add(q)
                except: continue

        found_count = 0
        for _, row in df.iterrows():
            # Apply same logic as controller.py to reconstruct the query string
            if ds['name'] == "truthful_qa": q_text = row.get('question')
            elif ds['name'] == "gsm8k": q_text = row.get('question')
            elif ds['name'] == "medmcqa":
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
            else: q_text = str(row.get(ds['key']))

            # Check if this exact string exists in our processed set
            if q_text in processed_set:
                found_count += 1
            else:
                # Fallback: substring match for formatting shifts
                snippet = q_text[:100]
                match = False
                for p_query in processed_set:
                    if snippet in p_query:
                        found_count += 1
                        match = True
                        break

        status = "✅ PERFECT" if found_count == len(df) else f"⚠️ MISSING {len(df)-found_count}"
        print(f"{ds['name']:<20} | {found_count:<10} | {status}")

if __name__ == "__main__":
    database_specific_audit()
