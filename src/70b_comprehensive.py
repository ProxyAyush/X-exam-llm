import pandas as pd
import json
import os
import re

RESULTS_DIR = "results"
BASELINE_DIR = "results_baseline"
DATA_DIR = "data"
MODEL = "llama-3.3-70b-versatile"
DATASETS = ["truthful_qa", "gsm8k", "medqa", "HaluEval", "medmcqa", "case_hold", "law_stack_exchange"]

def normalize_q(q):
    q = str(q).lower()
    prefixes = ["medical question: ", "legal context: ", "title: ", "question: ", "query: ", "context: "]
    for p in prefixes:
        if q.startswith(p): q = q[len(p):]
    if q.startswith('{'):
        try:
            import ast
            d = ast.literal_eval(q)
            for k in ['Question', 'question', 'knowledge', 'context', 'title']:
                if k in d: return re.sub(r'[^a-z0-9]', '', str(d[k]).lower())[:100]
            q = max([str(v) for v in d.values()], key=len)
        except: pass
    q = re.sub(r'[^a-z0-9]', '', q)
    return q[:100]

def get_gt_all():
    gt = {}
    # TruthfulQA
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "truthful_qa.parquet"))
        for _, row in df.iterrows():
            labels = list(row['mc1_targets']['labels'])
            choices = list(row['mc1_targets']['choices'])
            if 1 in labels: gt[normalize_q(row['question'])] = choices[labels.index(1)]
    except: pass

    # GSM8K
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "gsm8k.parquet"))
        for _, row in df.iterrows():
            match = re.search(r"#### (\d+)", row['answer'])
            gt[normalize_q(row['question'])] = match.group(1) if match else row['answer']
    except: pass

    # MedQA
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "medqa.parquet"))
        for _, row in df.iterrows():
            gt[normalize_q(row['data']['Question'])] = row['data']['Correct Answer']
    except: pass

    # HaluEval
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "HaluEval.parquet"))
        for _, row in df.iterrows():
            gt[normalize_q(row['question'])] = row['right_answer']
    except: pass

    # MedMCQA
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "medmcqa.parquet"))
        for _, row in df.iterrows():
            ans_map = {0: row['opa'], 1: row['opb'], 2: row['opc'], 3: row['opd']}
            gt[normalize_q(row['question'])] = ans_map.get(row['cop'], "unknown")
    except: pass

    # CaseHold
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "case_hold.parquet"))
        for _, row in df.iterrows():
            gt[normalize_q(row['context'])] = row['endings'][row['label']]
    except: pass

    # Law Stack Exchange
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, "law_stack_exchange.parquet"))
        for _, row in df.iterrows():
            gt[normalize_q(row['title'])] = row['text_label'] # Proxy for GT if available
    except: pass

    return gt

def is_correct(assertion, ground_truth, dataset):
    if not ground_truth: return False
    assertion, gt = str(assertion).lower(), str(ground_truth).lower()
    if dataset == "gsm8k":
        nums = re.findall(r"(\d+)", assertion)
        return nums[-1] == gt if nums else False
    return gt in assertion or assertion in gt

def analyze():
    gt_data = get_gt_all()
    rows = []

    for ds in DATASETS:
        x_path, b_path = os.path.join(RESULTS_DIR, ds, "results.jsonl"), os.path.join(BASELINE_DIR, ds, "results.jsonl")
        if not (os.path.exists(x_path) and os.path.exists(b_path)): continue
        
        x_m = [json.loads(l) for l in open(x_path) if json.loads(l).get('model_used') == MODEL]
        b_m = [json.loads(l) for l in open(b_path) if json.loads(l).get('model_used') == MODEL]
        b_map = {normalize_q(d['query']): d for d in b_m}

        n, b_acc, x_acc, caught, false_pos = 0, 0, 0, 0, 0
        for d_x in x_m:
            q_norm = normalize_q(d_x['query'])
            if q_norm in b_map and q_norm in gt_data:
                n += 1
                gt = gt_data[q_norm]
                ok_x = is_correct(d_x['final_assertion'], gt, ds)
                ok_b = is_correct(b_map[q_norm]['final_assertion'], gt, ds)
                rejected = "REJECT" in str(d_x.get('history', [{}])[-1].get('verdict', '')).upper()
                
                if ok_b: b_acc += 1
                if ok_x: x_acc += 1
                if not ok_b and rejected: caught += 1
                if ok_b and rejected: false_pos += 1

        if n > 0:
            rows.append({
                "Dataset": ds, "N": n,
                "Base Acc (%)": round(b_acc/n*100, 1),
                "X-Exam Acc (%)": round(x_acc/n*100, 1),
                "Catch Rate (%)": round(caught/(n-b_acc)*100, 1) if n > b_acc else 0,
                "False Pos Rate (%)": round(false_pos/b_acc*100, 1) if b_acc > 0 else 0
            })

    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    analyze()
