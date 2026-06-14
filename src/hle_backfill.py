"""One-shot backfill for the 2 missing HLE items (indices 1219, 1485)."""
import os
import json
import time
import re
import requests
import random
from datetime import datetime

MISSING_INDICES = [1219, 1485]
MAX_PROMPT_CHARS = 4000  # Truncate to avoid 413

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] BACKFILL: {msg}"
    with open("hle_action_logs.txt", "a") as f:
        f.write(entry + "\n")
    print(entry)

def get_api_keys():
    keys = [
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_API_KEY_AYUSHI"),
        os.environ.get("GROQ_API_KEY_AKAAKA")
    ]
    return [k for k in keys if k]

def call_api(prompt, system_prompt, api_keys, key_idx=0, model="llama-3.3-70b-versatile"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    # Truncate prompt if too long
    if len(prompt) > MAX_PROMPT_CHARS:
        log(f"Truncating prompt from {len(prompt)} to {MAX_PROMPT_CHARS} chars")
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n[...truncated for length...]"

    for attempt in range(5):
        headers = {
            "Authorization": f"Bearer {api_keys[key_idx % len(api_keys)]}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=90)
            if resp.status_code == 429:
                key_idx = (key_idx + 1) % len(api_keys)
                wait = (2 ** attempt) * 10 + random.random() * 5
                log(f"429 Rate limit, rotating key, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if resp.status_code == 413:
                log(f"413 even after truncation (attempt {attempt+1})")
                return None
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            log(f"Error attempt {attempt+1}: {e}")
            time.sleep((2 ** attempt) * 5)
    return None

def run_x_exam(query, gt_answer, api_keys, model="llama-3.3-70b-versatile"):
    # 1. Generator
    gen_res = call_api(query, "You are a PhD-level expert. Solve this question step-by-step. Enclose final answer in <assertion>.", api_keys, model=model)
    if not gen_res:
        return None
    
    match = re.search(r"<assertion>(.*?)</assertion>", gen_res, re.DOTALL | re.I)
    assertion = match.group(1).strip() if match else gen_res.strip()
    
    # 2. Cross-Examiner
    critique = call_api(
        f"Question: {query[:MAX_PROMPT_CHARS]}\nAssertion: {assertion}",
        "You are a ruthless, PhD-level adversarial auditor. Find any potential flaw, nuance, or edge case in the assertion.",
        api_keys, model=model
    )
    if not critique:
        return None
    
    # 3. Judge
    judge_res = call_api(
        f"Q: {query[:MAX_PROMPT_CHARS]}\nA: {assertion}\nCritique: {critique}",
        "Decide if the assertion is correct given the critique. Output <verdict>ACCEPT</verdict> or <verdict>REJECT</verdict>.",
        api_keys, model=model
    )
    if not judge_res:
        return None
    
    match_v = re.search(r"<verdict>(.*?)</verdict>", judge_res, re.I)
    verdict = match_v.group(1).upper() if match_v else "REJECT"

    return {
        "query": query,
        "gt": gt_answer,
        "final_assertion": assertion,
        "critique": critique,
        "verdict": verdict,
        "model_used": model,
        "timestamp": datetime.now().isoformat(),
        "mode": "x_exam_hle_backfill"
    }

def main():
    api_keys = get_api_keys()
    if not api_keys:
        log("CRITICAL: No API keys found. Cannot backfill.")
        return

    with open("data/hle.jsonl") as f:
        items = [json.loads(l) for l in f]

    log(f"Starting backfill for {len(MISSING_INDICES)} missing items: {MISSING_INDICES}")
    
    count = 0
    for idx in MISSING_INDICES:
        item = items[idx]
        query = item.get('question') or item.get('query') or ''
        answer = item.get('answer') or item.get('gt') or item.get('target') or ''
        
        log(f"Processing missing item {idx} (query length: {len(query)})...")
        result = run_x_exam(query, answer, api_keys)
        
        if result:
            result["original_index"] = idx
            with open("results/hle/results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
            count += 1
            log(f"SUCCESS: Item {idx} completed. Verdict: {result['verdict']}")
        else:
            log(f"FAILED: Could not process item {idx}")
        
        time.sleep(5)  # Breathing room between items
    
    # Update state
    with open("state_hle.json", "r") as f:
        state = json.load(f)
    state["total_processed"] = state.get("total_processed", 0) + count
    with open("state_hle.json", "w") as f:
        json.dump(state, f, indent=2)
    
    log(f"BACKFILL COMPLETE: {count}/{len(MISSING_INDICES)} items processed. Total now: {state['total_processed']}")

if __name__ == "__main__":
    main()
