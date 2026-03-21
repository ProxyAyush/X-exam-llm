import json
import os
from controller import XExamController

def backfill():
    # Load the missing indices
    with open("analysis/missing_indices.json", "r") as f:
        missing_map = json.load(f)

    # Initialize the standard X-Exam controller (Phase 3)
    controller = XExamController(state_path="state.json", results_dir="results")
    
    for ds_name, indices in missing_map.items():
        if not indices:
            continue
            
        print(f"\n>>> BACKFILLING: {ds_name} ({len(indices)} items)")
        
        # Find the dataset info in the controller state
        ds_info = next((d for d in controller.state['datasets'] if d['name'] == ds_name), None)
        if not ds_info: continue

        # Load the dataset (same sampling as controller)
        import pandas as pd
        df = pd.read_parquet(ds_info['file'])
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        
        dataset = df.to_dict(orient='records')

        for idx in indices:
            item = dataset[idx]
            query = item.get('question') or str(item)
            
            print(f"[{ds_name}] Processing Missing Item {idx}...")
            
            # Run the FULL X-Exam loop
            result = controller.run_x_exam_loop(query, controller.state["current_model"])
            
            if result:
                # Save to main results
                controller.save_result(ds_name, result, target_dir="results")
                print(f"  [SUCCESS] Item {idx} saved.")
            else:
                print(f"  [FAILED] API Error on item {idx}.")
                return # Stop and wait for next run if API fails

    print("\nBackfill operation complete.")

if __name__ == "__main__":
    backfill()
