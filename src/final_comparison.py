import pandas as pd
import json
import os

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
                
        # Find intersection with aggressive normalization
        def normalize(q):
            q = str(q).lower()
            # Remove common prefixes
            prefixes = ["medical question: ", "legal context: ", "title: ", "question: ", "query: ", "context: "]
            for p in prefixes:
                if q.startswith(p): q = q[len(p):]
            
            # Handle stringified dicts (like medqa, halueval)
            if q.startswith('{'):
                try:
                    import ast
                    d = ast.literal_eval(q)
                    # For HaluEval baseline, 'knowledge' is the key? No, wait.
                    # Looking at HaluEval output, it seems the results.jsonl (X-Exam) uses the QUESTION.
                    # The baseline used the KNOWLEDGE?
                    # Let's try to match by first few unique words or common content.
                    # Actually, if it's a dict, just take the longest string value as a proxy.
                    vals = [str(v) for v in d.values() if isinstance(v, str)]
                    if vals: q = max(vals, key=len)
                except: pass

            # Clean punctuation and whitespace
            import re
            q = re.sub(r'[^a-z0-9]', '', q)
            return q[:80] # Match first 80 alphanumeric chars

        xexam_norm = {}
        for q in xexam_items.keys():
            n = normalize(q)
            if n: xexam_norm[n] = q

        baseline_norm = {}
        for q in baseline_items.keys():
            n = normalize(q)
            if n: baseline_norm[n] = q
        
        common_norm_keys = set(xexam_norm.keys()) & set(baseline_norm.keys())
        total_common = len(common_norm_keys)
        
        if total_common == 0:
            print(f"  [!] No matching queries found for {ds}")
            # Print a sample for debugging
            sample_x = list(xexam_norm.keys())[:1]
            sample_b = list(baseline_norm.keys())[:1]
            print(f"      Sample X-Exam Norm: {sample_x}")
            print(f"      Sample Baseline Norm: {sample_b}")
            continue
            
        print(f"  [+] Matched {total_common} items for {ds}")

        # Analyze Judge impact
        xexam_rejects = 0
        for norm_q in common_norm_keys:
            orig_q = xexam_norm[norm_q]
            history = xexam_items[orig_q].get('history', [])
            if history:
                # Find the last verdict in history
                verdicts = [h.get('verdict') for h in history if 'verdict' in h]
                final_verdict = str(verdicts[-1]).upper() if verdicts else "ACCEPT"
                if "REJECT" in final_verdict:
                    xexam_rejects += 1
        
        comparison_stats.append({
            "Dataset": ds,
            "Common Items": total_common,
            "X-Exam Scrutiny Rate": (xexam_rejects / total_common) * 100
        })

    if not comparison_stats:
        print("No comparison data available yet.")
        return

    df = pd.DataFrame(comparison_stats)
    csv_path = os.path.join(OUTPUT_DIR, "final_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Final comparison CSV generated at {csv_path}")
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Dataset", y="X-Exam Scrutiny Rate")
        plt.title("X-Exam Adversarial Scrutiny Rate across Benchmarks")
        plt.ylabel("Percentage of Assertions Challenged (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "adversarial_impact.png")
        plt.savefig(plot_path)
        print(f"Visualization generated at {plot_path}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("Please ensure matplotlib, seaborn and pillow are installed for charts.")

if __name__ == "__main__":
    generate_comparison()
