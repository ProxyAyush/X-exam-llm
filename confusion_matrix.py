import json
import os
import pandas as pd
from collections import defaultdict
import re

DATASETS = ["truthful_qa", "gsm8k", "medqa", "HaluEval"]

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
    return q

def is_correct(assertion, ground_truth, dataset):
    if not ground_truth: return False
    assertion = str(assertion).lower()
    gt = str(ground_truth).lower()
    if dataset == "gsm8k":
        nums = re.findall(r"(\d+)", assertion)
        return nums[-1] == gt if nums else False
    if len(gt) <= 2:
        return re.search(r'\b' + re.escape(gt) + r'\b', assertion) is not None
    return gt in assertion or assertion in gt

gt = {}
for ds in DATASETS:
    try:
        df = pd.read_parquet(f"data/{ds}.parquet")
        for _, row in df.iterrows():
            if ds == "truthful_qa":
                choices = row['mc1_targets']['choices']
                labels = row['mc1_targets']['labels']
                correct = [choices[i] for i, l in enumerate(labels) if l == 1]
                gt[normalize_q(row['question'])] = correct[0] if correct else None
            elif ds == "gsm8k":
                match = re.search(r"#### (\d+)", row['answer'])
                gt[normalize_q(row['question'])] = match.group(1) if match else row['answer']
            elif ds == "medqa":
                gt[normalize_q(row['data']['Question'])] = row['data']['Correct Answer']
            elif ds == "HaluEval":
                gt[normalize_q(row['question'])] = row['right_answer']
    except Exception as e:
        print(f"Error loading {ds}: {e}")

matrix = defaultdict(lambda: {'Accept-Correct': 0, 'Accept-Wrong': 0, 'Reject-Correct': 0, 'Reject-Wrong': 0})

for ds in DATASETS:
    xpath = f"results/{ds}/results.jsonl"
    if not os.path.exists(xpath): continue
    
    xraw = [json.loads(l) for l in open(xpath)]
    
    for dx in xraw:
        q = normalize_q(dx['query'])
        if q in gt:
            g = gt[q]
            correct = is_correct(dx.get('generator_assertion', dx.get('final_assertion', '')), g, ds)
            verdict = dx.get('history', [{'verdict': 'ACCEPT'}])[-1].get('verdict', 'ACCEPT')
            
            if verdict == 'ACCEPT':
                if correct: matrix[ds]['Accept-Correct'] += 1
                else: matrix[ds]['Accept-Wrong'] += 1
            else:
                if correct: matrix[ds]['Reject-Correct'] += 1
                else: matrix[ds]['Reject-Wrong'] += 1

print("="*60)
print("JUDGE CONFUSION MATRIX (Adversarial Capitulation Analysis)")
print("="*60)
print(f"{'Dataset':<15} | {'Accept-Corr':<11} | {'Accept-Wrng':<11} | {'Reject-Corr':<11} | {'Reject-Wrng':<11} | {'Capitulation Rate':<17}")
print("-" * 88)
for ds, counts in matrix.items():
    ac = counts['Accept-Correct']
    aw = counts['Accept-Wrong']
    rc = counts['Reject-Correct']
    rw = counts['Reject-Wrong']
    total_rejected = rc + rw
    cap_rate = (rc / total_rejected * 100) if total_rejected > 0 else 0
    print(f"{ds:<15} | {ac:<11} | {aw:<11} | {rc:<11} | {rw:<11} | {cap_rate:.1f}%")
