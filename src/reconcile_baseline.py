import os
import json

def reconcile_baseline():
    state_path = "state_baseline.json"
    results_dir = "results_baseline"
    
    if not os.path.exists(state_path):
        print("No baseline state file found.")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    changed = False
    for ds in state['datasets']:
        clean_name = ds['name'].replace("/", "_")
        res_file = os.path.join(results_dir, clean_name, "results.jsonl")
        
        actual_count = 0
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                actual_count = sum(1 for line in f if line.strip())
        
        if ds['index'] != actual_count:
            print(f"Reconciling {ds['name']}: {ds['index']} -> {actual_count}")
            ds['index'] = actual_count
            changed = True

    if changed:
        # Check if we need to reset the current_dataset_idx
        # If the first dataset is not done, current_dataset_idx should be 0
        for i, ds in enumerate(state['datasets']):
            # This is a bit simplified, but let's just find the first non-full dataset
            # (Assuming we don't know the exact max items easily here, 
            # we rely on the runner to finish it)
            pass 
        
        # More importantly, if truthful_qa (index 0) is not actually at 817, 
        # we must set current_dataset_idx to 0.
        if state['datasets'][0]['index'] < 817:
            if state['current_dataset_idx'] != 0:
                print(f"Setting current_dataset_idx to 0 (TruthfulQA is not done).")
                state['current_dataset_idx'] = 0
                changed = True

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        print("Reconciliation complete.")
    else:
        print("All indices match actual counts.")

if __name__ == "__main__":
    reconcile_baseline()
