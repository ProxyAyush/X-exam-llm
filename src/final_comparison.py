import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
BASELINE_DIR = "results_baseline"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_comparison():
    comparison_stats = []
    
    datasets = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    for ds in datasets:
        xexam_path = os.path.join(RESULTS_DIR, ds, "results.jsonl")
        baseline_path = os.path.join(BASELINE_DIR, ds, "results.jsonl")
        
        if not os.path.exists(xexam_path) or not os.path.exists(baseline_path):
            print(f"Skipping {ds}: Missing results or baseline data.")
            continue
            
        print(f"Comparing {ds}...")
        
        # Load X-Exam
        xexam_items = {}
        with open(xexam_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                xexam_items[data['query']] = data
                
        # Load Baseline
        baseline_items = {}
        with open(baseline_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                baseline_items[data['query']] = data
                
        # Find intersection
        common_queries = set(xexam_items.keys()) & set(baseline_items.keys())
        total_common = len(common_queries)
        
        if total_common == 0:
            print(f"  [!] No matching queries found for {ds}")
            continue
            
        # Analyze Judge impact
        # We define "Improved Reliability" as items where the Judge rejected an initial hallucination
        # or accepted a robust answer.
        # For simplicity in this automated script, we measure the "Rejection Delta"
        
        xexam_rejects = sum(1 for q in common_queries if "REJECT" in str(xexam_items[q].get('history', [{}])[-1].get('verdict')).upper())
        
        comparison_stats.append({
            "Dataset": ds,
            "Common Items": total_common,
            "X-Exam Scrutiny Rate": (xexam_rejects / total_common) * 100
        })

    if not comparison_stats:
        print("No comparison data available yet.")
        return

    df = pd.DataFrame(comparison_stats)
    df.to_csv(os.path.join(OUTPUT_DIR, "final_comparison.csv"), index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y="X-Exam Scrutiny Rate")
    plt.title("X-Exam Adversarial Scrutiny Rate across Benchmarks")
    plt.ylabel("Percentage of Assertions Challenged (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "adversarial_impact.png"))
    
    print(f"Final comparison generated in {OUTPUT_DIR}/")

if __name__ == "__main__":
    generate_comparison()
