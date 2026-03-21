import json
import os
import pandas as pd

def generate_qa_summary(results_dir="results", output_file="analysis/all_qa_pairs.jsonl"):
    os.makedirs("analysis", exist_ok=True)
    all_pairs = []
    
    for dataset in sorted(os.listdir(results_dir)):
        res_path = os.path.join(results_dir, dataset, "results.jsonl")
        if not os.path.exists(res_path):
            continue
            
        print(f"Summarizing {dataset}...")
        with open(res_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_pairs.append({
                        "dataset": dataset,
                        "query": data.get("query"),
                        "final_assertion": data.get("final_assertion")
                    })
                except:
                    continue

    with open(output_file, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(f"Total pairs summarized: {len(all_pairs)}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_qa_summary()
