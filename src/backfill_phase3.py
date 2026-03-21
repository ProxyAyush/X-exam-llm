import json
import os
import time
import pandas as pd
from controller import XExamController

# FORCED MODEL: Identified as having the least data
TARGET_MODEL = "llama-3.3-70b-versatile"

def backfill():
    if not os.path.exists("analysis/missing_indices.json"):
        print("No missing indices found.")
        return

    with open("analysis/missing_indices.json", "r") as f:
        missing_map = json.load(f)

    controller = XExamController(state_path="state.json", results_dir="results")
    
    for ds_name, indices in missing_map.items():
        if not indices: continue
            
        print(f"\n>>> FORCED BACKFILL: {ds_name} ({len(indices)} items) using {TARGET_MODEL}")
        
        ds_info = next((d for d in controller.state['datasets'] if d['name'] == ds_name), None)
        if not ds_info: continue

        df = pd.read_parquet(ds_info['file'])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        dataset = df.to_dict(orient='records')

        for idx in indices:
            item = dataset[idx]
            
            # Use same query reconstruction logic as controller.py
            if ds_name == "truthful_qa": query = item.get('question')
            elif ds_name == "gsm8k": query = item.get('question')
            elif ds_name == "medmcqa":
                options = f"A) {item.get('opa')}\nB) {item.get('opb')}\nC) {item.get('opc')}\nD) {item.get('opd')}"
                query = f"{item.get('question')}\nOptions:\n{options}"
            elif ds_name == "medqa":
                data_dict = item.get('data', {})
                if isinstance(data_dict, str):
                    try: data_dict = json.loads(data_dict)
                    except: data_dict = {}
                q_text = data_dict.get('Question', 'N/A')
                opts = data_dict.get('Options', {})
                options_str = "\n".join([f"{k}) {v}" for k, v in opts.items()])
                query = f"Medical Question: {q_text}\nOptions:\n{options_str}"
            elif ds_name == "case_hold":
                opts = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(item.get('endings', []))])
                query = f"Legal Context: {item.get('context')}\nHolding Options:\n{opts}"
            elif ds_name == "law_stack_exchange":
                query = f"Title: {item.get('title')}\nQuestion: {item.get('body')}"
            else:
                query = str(item.get('question') or item)

            print(f"[{ds_name}] Processing Item {idx} with {TARGET_MODEL}...")
            
            result = controller.run_x_exam_loop(query, TARGET_MODEL, force_model=True)
            
            if result:
                controller.save_result(ds_name, result, target_dir="results")
                print(f"  [SUCCESS] Saved.")
            else:
                print(f"  [WAITING] API Keys exhausted. Sleeping 65s to reset RPM...")
                time.sleep(65)
                # Try one more time after sleep before giving up
                result = controller.run_x_exam_loop(query, TARGET_MODEL, force_model=True)
                if result:
                    controller.save_result(ds_name, result, target_dir="results")
                    print(f"  [SUCCESS] Saved after sleep.")
                else:
                    print(f"  [TERMINATING] Resource still unavailable.")
                    return 

    print("\nBalanced backfill complete.")

if __name__ == "__main__":
    backfill()
