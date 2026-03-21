import os
import json

def force_sync_phase3():
    state_path = "state.json"
    results_dir = "results"
    
    with open(state_path, 'r') as f:
        state = json.load(f)

    print(f"{'Dataset':<20} | {'Old Index':<10} | {'Actual Lines':<10}")
    print("-" * 50)

    for ds in state['datasets']:
        clean_name = ds['name'].replace("/", "_")
        res_file = os.path.join(results_dir, clean_name, "results.jsonl")
        
        actual_count = 0
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                actual_count = sum(1 for line in f if line.strip())
        
        print(f"{ds['name']:<20} | {ds['index']:<10} | {actual_count:<10}")
        ds['index'] = actual_count

    # Reset current_dataset_idx to the first one that isn't finished
    # We'll set it to 0 to be safe so it checks all of them
    state['current_dataset_idx'] = 0

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    print("\nState.json has been force-synced to reality.")

if __name__ == "__main__":
    force_sync_phase3()
