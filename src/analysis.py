import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import brier_score_loss

def calculate_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices of confidences that fall into the current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def analyze_results(results_dir="results", output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    
    for dataset_folder in os.listdir(results_dir):
        res_path = os.path.join(results_dir, dataset_folder, "results.jsonl")
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        data['dataset'] = dataset_folder
                        # Extract final verdict and confidence (placeholder for now)
                        verdicts = [h.get('verdict') for h in data.get('history', []) if 'verdict' in h]
                        data['final_verdict'] = verdicts[-1] if verdicts else "UNKNOWN"
                        data['num_iterations'] = len(verdicts)
                        
                        # Placeholder for ground truth (need to match with actual dataset ground truth)
                        # For now, assume verdict ACCEPT means correct (this is just for simulation/template)
                        data['is_correct'] = 1 if "ACCEPT" in data['final_verdict'].upper() else 0
                        # Placeholder confidence (need to extract from model if possible)
                        data['confidence'] = 0.9 if data['is_correct'] else 0.4
                        
                        all_data.append(data)
                    except Exception as e:
                        print(f"Error parsing line: {e}")
    
    if not all_data:
        print("No results found to analyze.")
        return

    df = pd.DataFrame(all_data)
    
    # 1. Basic Summary
    summary = df.groupby('dataset')['final_verdict'].value_counts().unstack().fillna(0)
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    
    # 2. Calibration Analysis (ECE)
    ece_scores = {}
    for ds in df['dataset'].unique():
        subset = df[df['dataset'] == ds]
        ece = calculate_ece(subset['confidence'].values, subset['is_correct'].values)
        ece_scores[ds] = ece
        
    ece_df = pd.DataFrame.from_dict(ece_scores, orient='index', columns=['ECE'])
    ece_df.to_csv(os.path.join(output_dir, "ece_scores.csv"))
    
    # 3. Visualization: Verdict Distribution
    plt.figure(figsize=(12, 6))
    summary.plot(kind='bar', stacked=True)
    plt.title("X-Exam Verdict Distribution across Datasets")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "verdict_dist.png"))
    plt.close()

    # 4. Visualization: Iteration Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="num_iterations", hue="dataset", multiple="dodge", shrink=.8)
    plt.title("Number of Adversarial Iterations before Resolution")
    plt.savefig(os.path.join(output_dir, "iterations_dist.png"))
    plt.close()

    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    analyze_results()
