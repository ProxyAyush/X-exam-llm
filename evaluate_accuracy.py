import pandas as pd
import json
import os
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
    return q[:100]

def is_correct(assertion, ground_truth, dataset):
    if not ground_truth: return False
    assertion = str(assertion).lower()
    gt = str(ground_truth).lower()
    if dataset == "gsm8k":
        nums = re.findall(r"(\d+)", assertion)
        return nums[-1] == gt if nums else False
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
        pass

results = []
for ds in DATASETS:
    xpath = f"results/{ds}/results.jsonl"
    bpath = f"results_baseline/{ds}/results.jsonl"
    if not os.path.exists(xpath) or not os.path.exists(bpath): continue
    
    xraw = [json.loads(l) for l in open(xpath)]
    braw = [json.loads(l) for l in open(bpath)]
    
    for model in ["llama-3.3-70b-versatile", "qwen/qwen3-32b"]:
        mx = [d for d in xraw if d.get('model_used') == model]
        mb = [d for d in braw if d.get('model_used') == model]
        bmap = {normalize_q(d['query']): d for d in mb}
        
        x_correct = 0
        b_correct = 0
        total = 0
        
        for dx in mx:
            q = normalize_q(dx['query'])
            if q in bmap and q in gt:
                total += 1
                g = gt[q]
                if is_correct(dx['final_assertion'], g, ds): x_correct += 1
                if is_correct(bmap[q]['final_assertion'], g, ds): b_correct += 1
                
        if total > 0:
            results.append({
                "Dataset": ds, "Model": model.split('-')[0] if 'llama' in model else 'qwen3',
                "N": total,
                "Base_Acc": round(b_correct/total*100, 2),
                "XExam_Acc": round(x_correct/total*100, 2),
                "Delta": round((x_correct - b_correct)/total*100, 2)
            })

print(pd.DataFrame(results).to_string())
