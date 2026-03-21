import json
import os
from controller import XExamController

# FORCED MODEL: Identified as having the least data (1,350 items)
TARGET_MODEL = "llama-3.3-70b-versatile"

def backfill():
    with open("analysis/missing_indices.json", "r") as f:
        missing_map = json.load(f)

    controller = XExamController(state_path="state.json", results_dir="results")
    
    for ds_name, indices in missing_map.items():
        if not indices: continue
            
        print(f"\n>>> FORCED BACKFILL: {ds_name} using {TARGET_MODEL}")
        
        ds_info = next((d for d in controller.state['datasets'] if d['name'] == ds_name), None)
        if not ds_info: continue

        import pandas as pd
        df = pd.read_parquet(ds_info['file'])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        dataset = df.to_dict(orient='records')

        for idx in indices:
            item = dataset[idx]
            query = item.get('question') or str(item)
            
            print(f"[{ds_name}] Processing Item {idx} with {TARGET_MODEL}...")
            
            # We force the model here
            result = controller.run_x_exam_loop(query, TARGET_MODEL, force_model=True)
            
            if result:
                controller.save_result(ds_name, result, target_dir="results")
                print(f"  [SUCCESS] Saved.")
            else:
                print(f"  [FAILED] API Error or Exhaustion.")
                return 

    print("\nBalanced backfill complete.")

if __name__ == "__main__":
    backfill()
