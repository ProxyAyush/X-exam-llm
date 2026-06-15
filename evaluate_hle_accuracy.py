import json

correct = 0
total = 0

with open("results/hle/results.jsonl") as f:
    for line in f:
        data = json.loads(line)
        gt = str(data.get('gt', '')).strip().lower()
        ans = str(data.get('final_assertion', '')).strip().lower()
        
        if not gt:
            continue
            
        total += 1
        
        # Simple evaluation:
        # If gt is a single letter (multiple choice)
        if len(gt) == 1 and len(ans) >= 1:
            # Often the model might say "The answer is A" or "<assertion>A</assertion>"
            # Just checking if the gt is the only letter in the answer, or if the answer starts/ends with it.
            # A more robust check:
            if ans == gt or f" {gt}" in ans or f"{gt} " in ans or f"({gt})" in ans:
                correct += 1
        else:
            # If it's a phrase or number
            if gt in ans or ans in gt:
                correct += 1

print(f"HLE X-Exam Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
