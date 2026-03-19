import json
import os
import pandas as pd

def mine_discoveries(results_dir="results", output_file="analysis/mined_trajectories.md"):
    trajectories = []
    
    for dataset in os.listdir(results_dir):
        res_path = os.path.join(results_dir, dataset, "results.jsonl")
        if not os.path.exists(res_path):
            continue
            
        with open(res_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                history = data.get('history', [])
                
                # We are looking for "interesting" cases:
                # 1. Successful Corrections (Reject -> Accept)
                # 2. Stubborn Hallucinations (Reject -> Max Iterations)
                
                verdicts = [h.get('verdict') for h in history if 'verdict' in h]
                if any(v and "REJECT" in v.upper() for v in verdicts):
                    trajectories.append({
                        "dataset": dataset,
                        "query": data['query'],
                        "initial_assertion": history[0].get('assertion', 'N/A'),
                        "critique": history[1].get('critique', 'N/A') if len(history) > 1 else 'N/A',
                        "final_verdict": verdicts[-1],
                        "iterations": len(verdicts)
                    })

    if not trajectories:
        print("No adversarial correction trajectories found yet.")
        return

    # Generate Markdown Report
    with open(output_file, 'w') as f:
        f.write("# Mined Adversarial Trajectories\n\n")
        f.write(f"Found **{len(trajectories)}** instances where the Cross-Examiner identified potential flaws.\n\n")
        
        for i, t in enumerate(trajectories[:10]): # Show top 10 for now
            f.write(f"### Discovery Case {i+1} [{t['dataset']}]\n")
            f.write(f"**Query:** {t['query']}\n\n")
            f.write(f"**Final Verdict:** {t['final_verdict']} (after {t['iterations']} rounds)\n\n")
            f.write("<details>\n<summary>View Adversarial Critique</summary>\n\n")
            f.write(f"**Initial Assertion:** {t['initial_assertion']}\n\n")
            f.write(f"**Adversarial Critique:** {t['critique']}\n")
            f.write("\n</details>\n\n---\n")

    print(f"Discovery report generated: {output_file}")

if __name__ == "__main__":
    mine_discoveries()
