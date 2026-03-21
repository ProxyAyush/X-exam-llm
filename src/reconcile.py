import json
import os
import pandas as pd

STATE_FILE = "state.json"
RESULTS_DIR = "results"
MAX_ITEMS = 2000

def reconcile():
    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    print("--- X-Exam Progress Reconciliation ---")
    
    any_changed = False
    
    for i, ds in enumerate(state["datasets"]):
        name = ds["name"]
        res_path = os.path.join(RESULTS_DIR, name.replace("/", "_"), "results.jsonl")
        
        if os.path.exists(res_path):
            # Read and deduplicate
            unique_results = {}
            with open(res_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Use query as the unique key
                        query = data.get("query")
                        if query:
                            unique_results[query] = data
                    except:
                        continue
            
            actual_count = len(unique_results)
            print(f"Dataset: {name}")
            print(f"  - Logged Index: {ds['index']}")
            print(f"  - Actual Unique Items: {actual_count}")
            
            # Update state if different
            if ds["index"] != actual_count:
                print(f"  [!] Reconciling index: {ds['index']} -> {actual_count}")
                ds["index"] = actual_count
                any_changed = True
            
            # Rewrite file to deduplicate and clean
            with open(res_path, "w") as f:
                for data in unique_results.values():
                    f.write(json.dumps(data) + "\n")
        else:
            print(f"Dataset: {name} - No results file found yet.")
            if ds["index"] != 0:
                print(f"  [!] Resetting index to 0")
                ds["index"] = 0
                any_changed = True

    # Recalculate current_dataset_idx
    # Find the first dataset that is not finished
    new_idx = len(state["datasets"]) # Default to 'done'
    for i, ds in enumerate(state["datasets"]):
        # Check actual total size of dataset
        try:
            df = pd.read_parquet(ds["file"])
            total_available = len(df)
        except:
            total_available = 0
            
        target = min(total_available, MAX_ITEMS)
        if ds["index"] < target:
            new_idx = i
            break
    
    if state["current_dataset_idx"] != new_idx:
        print(f"Updating current_dataset_idx: {state['current_dataset_idx']} -> {new_idx}")
        state["current_dataset_idx"] = new_idx
        any_changed = True

    if any_changed:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print("\nReconciliation complete. state.json updated.")
    else:
        print("\nNo changes needed. state.json is already in sync with results.")

if __name__ == "__main__":
    reconcile()
