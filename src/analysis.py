import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_results(results_dir="results", output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    
    for dataset in os.listdir(results_dir):
        res_path = os.path.join(results_dir, dataset, "results.jsonl")
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    data['dataset'] = dataset
                    all_data.append(data)
    
    if not all_data:
        print("No results found to analyze.")
        return

    df = pd.DataFrame(all_data)
    # Placeholder for actual analysis (e.g. ECE, Brier Score, etc.)
    # For now, just generate a simple summary
    summary = df.groupby('dataset')['judge_verdict'].value_counts().unstack().fillna(0)
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    summary.plot(kind='bar', stacked=True)
    plt.title("X-Exam Verdict Distribution across Datasets")
    plt.savefig(os.path.join(output_dir, "verdict_dist.png"))
    plt.close()

if __name__ == "__main__":
    analyze_results()
