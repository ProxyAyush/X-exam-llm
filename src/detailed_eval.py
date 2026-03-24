import pandas as pd
import json
import os
import re

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
        # mc1_targets: {'choices': [...], 'labels': [1, 0, 0...]}
        choices = row['mc1_targets']['choices']
        labels = row['mc1_targets']['labels']
        correct = [choices[i] for i, l in enumerate(labels) if l == 1]
        gt[q_norm] = correct[0] if correct else None

    # GSM8K
    df = pd.read_parquet(os.path.join(DATA_DIR, "gsm8k.parquet"))
    for _, row in df.iterrows():
        q_norm = normalize_q(row['question'])
        # Extract number from "#### 42"
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
        # Match last number
        nums = re.findall(r"(\d+)", assertion)
        return nums[-1] == gt if nums else False
    else:
        # Check if GT is in assertion or vice versa
        return gt in assertion or assertion in gt

def evaluate():
    gt_data = get_gt()
    stats = []

    for ds in DATASETS:
        x_path = os.path.join(RESULTS_DIR, ds, "results.jsonl")
        b_path = os.path.join(BASELINE_DIR, ds, "results.jsonl")
        
        if not os.path.exists(x_path) or not os.path.exists(b_path): continue

        x_data = []
        with open(x_path, 'r') as f:
            for line in f: x_data.append(json.loads(line))
        
        b_data = []
        with open(b_path, 'r') as f:
            for line in f: b_data.append(json.loads(line))

        for model in MODELS:
            m_x = [d for d in x_data if d.get('model_used') == model]
            m_b = [d for d in b_data if d.get('model_used') == model]
            
            # Map by norm query
            b_map = {normalize_q(d['query']): d for d in m_b}
            
            correct_x = 0
            correct_b = 0
            total = 0
            hal_caught = 0
            
            for d_x in m_x:
                q_norm = normalize_q(d_x['query'])
                if q_norm in b_map and q_norm in gt_data:
                    total += 1
                    d_b = b_map[q_norm]
                    gt = gt_data[q_norm]
                    
                    ok_x = is_correct(d_x['final_assertion'], gt, ds)
                    ok_b = is_correct(d_b['final_assertion'], gt, ds)
                    
                    if ok_x: correct_x += 1
                    if ok_b: correct_b += 1
                    
                    # Hallucination caught: Baseline was WRONG, but X-Exam REJECTED it
                    # (Simplified: X-Exam history has REJECT)
                    verdicts = [h.get('verdict') for h in d_x.get('history', []) if 'verdict' in h]
                    if not ok_b and verdicts and "REJECT" in str(verdicts[-1]).upper():
                        hal_caught += 1

            if total > 0:
                stats.append({
                    "Dataset": ds,
                    "Model": model,
                    "Total": total,
                    "Baseline Acc": (correct_b / total) * 100,
                    "X-Exam Acc": (correct_x / total) * 100,
                    "Hallucinations Caught": hal_caught,
                    "Catch Rate (%)": (hal_caught / (total - correct_b)) * 100 if (total - correct_b) > 0 else 0
                })

    df_res = pd.DataFrame(stats)
    print(df_res.to_string(index=False))
    df_res.to_csv("analysis/detailed_model_comparison.csv", index=False)

if __name__ == "__main__":
    evaluate()
