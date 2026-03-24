import pandas as pd
import json
import os
import re
import math

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
    df = pd.read_parquet(os.path.join(DATA_DIR, "truthful_qa.parquet"))
    for _, row in df.iterrows():
        q_norm = normalize_q(row['question'])
        choices = row['mc1_targets']['choices']
        labels = row['mc1_targets']['labels']
        correct = [choices[i] for i, l in enumerate(labels) if l == 1]
        gt[q_norm] = correct[0] if correct else None
    df = pd.read_parquet(os.path.join(DATA_DIR, "gsm8k.parquet"))
    for _, row in df.iterrows():
        q_norm = normalize_q(row['question'])
        match = re.search(r"#### (\d+)", row['answer'])
        gt[q_norm] = match.group(1) if match else row['answer']
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

def chi_squared_p(table):
    # table = [[AcceptCor, AcceptWrong], [RejectCor, RejectWrong]]
    a, b = table[0][0], table[0][1]
    c, d = table[1][0], table[1][1]
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    n = a + b + c + d
    if n == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 1.0
    
    expected = [
        [row1 * col1 / n, row1 * col2 / n],
        [row2 * col1 / n, row2 * col2 / n]
    ]
    
    chi_sq = 0
    for i in range(2):
        for j in range(2):
            o = [ [a, b], [c, d] ][i][j]
            e = expected[i][j]
            chi_sq += (o - e)**2 / e
            
    # For df=1, p-value from chi-square
    # Simple approx for p-value from chi-square df=1
    # p = exp(-chi_sq / 2) * (some series) - we can use a simpler threshold
    return chi_sq

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
            
            table_fish = [[0, 0], [0, 0]]
            total = 0
            for d_x in m_x:
                q_norm = normalize_q(d_x['query'])
                if q_norm in b_map and q_norm in gt_data:
                    total += 1
                    gt = gt_data[q_norm]
                    ok_b = is_correct(b_map[q_norm]['final_assertion'], gt, ds)
                    verdicts = [h.get('verdict') for h in d_x.get('history', []) if 'verdict' in h]
                    rejected = "REJECT" in str(verdicts[-1]).upper() if verdicts else False
                    
                    if ok_b: # Baseline CORRECT
                        if not rejected: table_fish[0][0] += 1
                        else: table_fish[1][0] += 1
                    else: # Baseline WRONG
                        if not rejected: table_fish[0][1] += 1
                        else: table_fish[1][1] += 1

            if total > 5:
                chi_val = chi_squared_p(table_fish)
                # Chi-sq > 3.84 is p < 0.05
                # Chi-sq > 6.63 is p < 0.01
                # Chi-sq > 10.83 is p < 0.001
                sig = "NO"
                if chi_val > 10.83: sig = "P < 0.001 (EXTREMELY SIG)"
                elif chi_val > 6.63: sig = "P < 0.01 (HIGHLY SIG)"
                elif chi_val > 3.84: sig = "P < 0.05 (SIG)"
                
                results.append({
                    "Dataset": ds, "Model": model, "N": total,
                    "Chi-Square": round(chi_val, 2),
                    "Targeted Significance": sig
                })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_tests()
