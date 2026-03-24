import pandas as pd
import json
import os
import re
from scipy.stats import fisher_exact, binom

RESULTS_DIR = "results"
BASELINE_DIR = "results_baseline"
DATA_DIR = "data"

MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen/qwen3-32b"]
DATASETS = ["truthful_qa", "gsm8k", "medqa"]

def normalize_q(q):
    q = str(q).lower()
    prefixes = ["medical question: ", "legal context: ", "title: ", "question: ", "query: ", "context: "]
    for p in prefixes:
        if q.startswith(p): q = q[len(p):]
    if q.startswith('{'):
        try:
            import ast
            d = ast.literal_eval(q)
            if 'Question' in d: q = d['Question']
            elif 'question' in d: q = d['question']
            else: q = max([str(v) for v in d.values()], key=len)
        except: pass
    q = re.sub(r'[^a-z0-9]', '', q)
    return q[:100]

def get_gt():
    gt = {}
    # TruthfulQA
    df = pd.read_parquet(os.path.join(DATA_DIR, "truthful_qa.parquet"))
    for _, row in df.iterrows():
        q_norm = normalize_q(row['question'])
        choices = row['mc1_targets']['choices']
        labels = row['mc1_targets']['labels']
        correct = [choices[i] for i, l in enumerate(labels) if l == 1]
        gt[q_norm] = correct[0] if correct else None

    # GSM8K
    df = pd.read_parquet(os.path.join(DATA_DIR, "gsm8k.parquet"))
    for _, row in df.iterrows():
        q_norm = normalize_q(row['question'])
        match = re.search(r"#### (\d+)", row['answer'])
        gt[q_norm] = match.group(1) if match else row['answer']

    # MedQA
    df = pd.read_parquet(os.path.join(DATA_DIR, "medqa.parquet"))
    for _, row in df.iterrows():
        d = row['data']
        q_norm = normalize_q(d['Question'])
        gt[q_norm] = d['Correct Answer']
    
    return gt

def is_correct(assertion, ground_truth, dataset):
    if not ground_truth: return False
    assertion = str(assertion).lower()
    gt = str(ground_truth).lower()
    if dataset == "gsm8k":
        nums = re.findall(r"(\d+)", assertion)
        return nums[-1] == gt if nums else False
    return gt in assertion or assertion in gt

def mcnemar_exact(table):
    b = table[0][1]
    c = table[1][0]
    n = b + c
    if n == 0: return 1.0
    p_val = 2 * binom.cdf(min(b, c), n, 0.5)
    return min(p_val, 1.0)

def run_tests():
    gt_data = get_gt()
    results = []

    for ds in DATASETS:
        x_path = os.path.join(RESULTS_DIR, ds, "results.jsonl")
        b_path = os.path.join(BASELINE_DIR, ds, "results.jsonl")
        if not (os.path.exists(x_path) and os.path.exists(b_path)): continue

        x_raw = [json.loads(l) for l in open(x_path)]
        b_raw = [json.loads(l) for l in open(b_path)]

        for model in MODELS:
            m_x = [d for d in x_raw if d.get('model_used') == model]
            m_b = [d for d in b_raw if d.get('model_used') == model]
            b_map = {normalize_q(d['query']): d for d in m_b}
            
            table_acc = [[0, 0], [0, 0]]
            table_fish = [[0, 0], [0, 0]]

            total = 0
            for d_x in m_x:
                q_norm = normalize_q(d_x['query'])
                if q_norm in b_map and q_norm in gt_data:
                    total += 1
                    gt = gt_data[q_norm]
                    ok_x = is_correct(d_x['final_assertion'], gt, ds)
                    ok_b = is_correct(b_map[q_norm]['final_assertion'], gt, ds)
                    
                    if ok_b and ok_x: table_acc[0][0] += 1
                    elif ok_b and not ok_x: table_acc[0][1] += 1
                    elif not ok_b and ok_x: table_acc[1][0] += 1
                    else: table_acc[1][1] += 1
                    
                    verdicts = [h.get('verdict') for h in d_x.get('history', []) if 'verdict' in h]
                    rejected = "REJECT" in str(verdicts[-1]).upper() if verdicts else False
                    
                    if ok_b: # Baseline was CORRECT
                        if not rejected: table_fish[0][0] += 1 # Accept Correct
                        else: table_fish[1][0] += 1            # Reject Correct
                    else: # Baseline was WRONG
                        if not rejected: table_fish[0][1] += 1 # Accept Wrong
                        else: table_fish[1][1] += 1            # Reject Wrong

            if total > 5:
                p_acc = mcnemar_exact(table_acc)
                odds, p_fish = fisher_exact(table_fish)
                
                results.append({
                    "Dataset": ds, "Model": model, "N": total,
                    "P-Value (Acc Shift)": p_acc,
                    "P-Value (Targeted Reject)": p_fish,
                    "Significance": "YES" if p_fish < 0.05 else "NO"
                })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_tests()
