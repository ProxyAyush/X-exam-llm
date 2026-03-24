import pandas as pd
import json
import os
import re
import math

RESULTS_DIR = "results"
BASELINE_DIR = "results_baseline"
DATA_DIR = "data"

MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen/qwen3-32b"]
DATASETS = ["truthful_qa", "gsm8k", "medqa", "HaluEval", "medmcqa", "case_hold"]

def normalize_q(q):
    q = str(q).lower()
    prefixes = ["medical question: ", "legal context: ", "title: ", "question: ", "query: ", "context: "]
    for p in prefixes:
        if q.startswith(p): q = q[len(p):]
    if q.startswith('{'):
        try:
            import ast
            d = ast.literal_eval(q)
            for k in ['Question', 'question', 'knowledge', 'context']:
                if k in d: return re.sub(r'[^a-z0-9]', '', str(d[k]).lower())[:100]
            q = max([str(v) for v in d.values()], key=len)
        except: pass
    q = re.sub(r'[^a-z0-9]', '', q)
    return q[:100]

def get_gt():
    gt = {}
    # TruthfulQA
    df = pd.read_parquet(os.path.join(DATA_DIR, "truthful_qa.parquet"))
    for _, row in df.iterrows():
        labels = list(row['mc1_targets']['labels'])
        choices = list(row['mc1_targets']['choices'])
        gt_val = None
        if 1 in labels:
            gt_val = choices[labels.index(1)]
        gt[normalize_q(row['question'])] = gt_val

    # GSM8K
    df = pd.read_parquet(os.path.join(DATA_DIR, "gsm8k.parquet"))
    for _, row in df.iterrows():
        match = re.search(r"#### (\d+)", row['answer'])
        gt[normalize_q(row['question'])] = match.group(1) if match else row['answer']

    # MedQA
    df = pd.read_parquet(os.path.join(DATA_DIR, "medqa.parquet"))
    for _, row in df.iterrows():
        gt[normalize_q(row['data']['Question'])] = row['data']['Correct Answer']
    
    # HaluEval
    df = pd.read_parquet(os.path.join(DATA_DIR, "HaluEval.parquet"))
    for _, row in df.iterrows():
        # HaluEval is tricky, often based on 'knowledge' or 'question'
        gt[normalize_q(row['question'])] = row['right_answer']
        gt[normalize_q(row['knowledge'])] = row['right_answer']

    # MedMCQA
    df = pd.read_parquet(os.path.join(DATA_DIR, "medmcqa.parquet"))
    for _, row in df.iterrows():
        ans_map = {0: row['opa'], 1: row['opb'], 2: row['opc'], 3: row['opd']}
        gt[normalize_q(row['question'])] = ans_map.get(row['cop'], "unknown")

    # CaseHold
    df = pd.read_parquet(os.path.join(DATA_DIR, "case_hold.parquet"))
    for _, row in df.iterrows():
        # endings is a list, label is index
        try:
            ans = row['endings'][row['label']]
            gt[normalize_q(row['context'])] = ans
        except: pass

    return gt

def is_correct(assertion, ground_truth, dataset):
    if not ground_truth: return False
    assertion, gt = str(assertion).lower(), str(ground_truth).lower()
    if dataset == "gsm8k":
        nums = re.findall(r"(\d+)", assertion)
        return nums[-1] == gt if nums else False
    return gt in assertion or assertion in gt

def chi_squared_p(table):
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
    row1, row2, col1, col2, n = a+b, c+d, a+c, b+d, a+b+c+d
    if n == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0: return 0
    expected = [[row1*col1/n, row1*col2/n], [row2*col1/n, row2*col2/n]]
    chi_sq = 0
    for i in range(2):
        for j in range(2):
            o = [[a, b], [c, d]][i][j]
            chi_sq += (o - expected[i][j])**2 / expected[i][j]
    return chi_sq

def run_tests():
    gt_data = get_gt()
    model_aggregates = {m: [[0, 0], [0, 0]] for m in MODELS}

    for ds in DATASETS:
        x_path, b_path = os.path.join(RESULTS_DIR, ds, "results.jsonl"), os.path.join(BASELINE_DIR, ds, "results.jsonl")
        if not (os.path.exists(x_path) and os.path.exists(b_path)): continue
        x_raw, b_raw = [json.loads(l) for l in open(x_path)], [json.loads(l) for l in open(b_path)]

        for model in MODELS:
            m_x = [d for d in x_raw if d.get('model_used') == model]
            b_map = {normalize_q(d['query']): d for d in b_raw if d.get('model_used') == model}
            for d_x in m_x:
                q_norm = normalize_q(d_x['query'])
                if q_norm in b_map and q_norm in gt_data:
                    gt = gt_data[q_norm]
                    ok_b = is_correct(b_map[q_norm]['final_assertion'], gt, ds)
                    rejected = "REJECT" in str(d_x.get('history', [{}])[-1].get('verdict', '')).upper()
                    if ok_b: # Baseline CORRECT
                        if not rejected: model_aggregates[model][0][0] += 1
                        else: model_aggregates[model][1][0] += 1
                    else: # Baseline WRONG
                        if not rejected: model_aggregates[model][0][1] += 1
                        else: model_aggregates[model][1][1] += 1

    final_results = []
    for model, table in model_aggregates.items():
        n = sum(table[0]) + sum(table[1])
        if n > 0:
            chi_val = chi_squared_p(table)
            sig = "NO"
            if chi_val > 10.83: sig = "P < 0.001 (EXTREMELY SIG)"
            elif chi_val > 6.63: sig = "P < 0.01 (HIGHLY SIG)"
            elif chi_val > 3.84: sig = "P < 0.05 (SIG)"
            final_results.append({"Model": model, "N": n, "Chi-Square": round(chi_val, 2), "Targeted Significance": sig})

    print(pd.DataFrame(final_results).to_string(index=False))

if __name__ == "__main__":
    run_tests()
