import os
import json
import time
import argparse
import re
import pandas as pd
from datetime import datetime
from groq import Groq
from tqdm import tqdm

RATE_LIMITS = {
    "llama-3.3-70b-versatile": {"rpm": 30, "tpm": 12000, "tpd": 100000},
    "llama-3.1-8b-instant": {"rpm": 30, "tpm": 12000, "tpd": 500000},
    "qwen/qwen3-32b": {"rpm": 60, "tpm": 6000, "tpd": 500000}
}

MODEL_FALLBACK_LIST = ["llama-3.3-70b-versatile", "qwen/qwen3-32b", "llama-3.1-8b-instant"]

class XExamController:
    def __init__(self, state_path=None, results_dir="results"):
        self.state_path = state_path if state_path else "state.json"
        self.results_dir = results_dir
        self.load_state()
        self.total_seconds_at_start = self.state.get("total_compute_seconds", 0)
        
        self.api_keys = [os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_AYUSHI"), os.environ.get("GROQ_API_KEY_AKAAKA")]
        self.api_keys = [k for k in self.api_keys if k]
        self.current_key_idx = 0
        self.client = Groq(api_key=self.api_keys[self.current_key_idx])
        
        self.request_times = {model: [] for model in RATE_LIMITS}
        self.exhausted_models = set()
        self.start_time = time.time()
        self.max_runtime_seconds = 3000 
        self.model_mapping = {}

    def load_model_mapping(self, dataset_name):
        self.model_mapping = {}
        res_path = os.path.join("results", dataset_name.replace("/", "_"), "results.jsonl")
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        query, model = data.get("query"), data.get("model_used")
                        if query and model: self.model_mapping[query] = model
                    except: continue
        print(f"DEBUG: Loaded {len(self.model_mapping)} model mappings for {dataset_name}")

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f: self.state = json.load(f)
        else:
            self.state = {
                "current_dataset_idx": 0, 
                "current_model": "llama-3.3-70b-versatile", 
                "total_compute_seconds": 0, 
                "datasets": [
                    {"name": "truthful_qa", "file": "data/truthful_qa.parquet", "index": 0},
                    {"name": "HaluEval", "file": "data/HaluEval.parquet", "index": 0},
                    {"name": "gsm8k", "file": "data/gsm8k.parquet", "index": 0},
                    {"name": "medmcqa", "file": "data/medmcqa.parquet", "index": 0},
                    {"name": "medqa", "file": "data/medqa.parquet", "index": 0},
                    {"name": "case_hold", "file": "data/case_hold.parquet", "index": 0},
                    {"name": "law_stack_exchange", "file": "data/law_stack_exchange.parquet", "index": 0}
                ]
            }

    def save_state(self):
        self.state["total_compute_seconds"] = self.total_seconds_at_start + (time.time() - self.start_time)
        with open(self.state_path, 'w') as f: json.dump(self.state, f, indent=2)
        print(f"DEBUG: Progress Saved: {self.state_path}")

    def rotate_key(self):
        self.current_key_idx += 1
        if self.current_key_idx < len(self.api_keys):
            print(f"DEBUG: Rotating to Key Index {self.current_key_idx}")
            self.client = Groq(api_key=self.api_keys[self.current_key_idx])
            return True
        return False

    def call_groq(self, model, prompt, system_prompt="You are a helpful assistant.", force_model=False):
        limits = RATE_LIMITS.get(model)
        if limits:
            now = time.time()
            self.request_times[model] = [t for t in self.request_times[model] if now - t < 60]
            if len(self.request_times[model]) >= limits["rpm"]:
                time.sleep(30)
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                model=model,
            )
            self.request_times[model].append(time.time())
            return chat_completion.choices[0].message.content
        except Exception as e:
            err_msg = str(e).lower()
            print(f"API ERROR ({model}): {e}")
            if "rate limit" in err_msg or "429" in err_msg:
                if self.rotate_key():
                    return self.call_groq(model, prompt, system_prompt, force_model=force_model)
                elif not force_model:
                    self.exhausted_models.add(model)
                    for m in MODEL_FALLBACK_LIST:
                        if m not in self.exhausted_models:
                            print(f"DEBUG: Switching to Fallback: {m}")
                            self.state["current_model"] = m
                            return self.call_groq(m, prompt, system_prompt)
            return None

    def run_x_exam_loop(self, query, model, iterations=1, baseline=False, force_model=False):
        # 1. Generator
        gen_system = "You are an expert domain solver. Provide a step-by-step solution. You must enclose your final assertion within <assertion> tags."
        res = self.call_groq(model, query, gen_system, force_model=force_model)
        if not res: return None
        
        match = re.search(r"<assertion>(.*?)</assertion>", res, re.DOTALL | re.IGNORECASE)
        assertion = match.group(1).strip() if match else res.strip()
        
        if baseline:
            return {
                "query": query,
                "final_assertion": assertion,
                "history": [{"role": "generator", "content": res, "assertion": assertion}],
                "model_used": model,
                "timestamp": datetime.now().isoformat(),
                "mode": "baseline"
            }

        # 2. Adversary (X-Exam Mode Only)
        history = [{"role": "generator", "content": res, "assertion": assertion}]
        exam_system = "You are a ruthless, adversarial cross-examiner. Identify flaws in the assertion."
        critique = self.call_groq(model, f"Query: {query}\nAssertion: {assertion}", exam_system, force_model=force_model)
        if not critique: return None

        # 3. Judge
        judge_system = "Output <verdict>ACCEPT</verdict> or <verdict>REJECT</verdict>."
        verdict_raw = self.call_groq(model, f"Query: {query}\nAssertion: {assertion}\nCritique: {critique}", judge_system, force_model=force_model)
        if not verdict_raw: return None

        match_v = re.search(r"<verdict>(.*?)</verdict>", verdict_raw, re.DOTALL | re.IGNORECASE)
        verdict = match_v.group(1).strip() if match_v else "REJECT"
        history.append({"iteration": 0, "critique": critique, "verdict": verdict})

        return {
            "query": query,
            "final_assertion": assertion,
            "history": history,
            "model_used": model,
            "timestamp": datetime.now().isoformat(),
            "mode": "x_exam"
        }

    def process_all(self, baseline=False):
        target_results_dir = self.results_dir if not baseline else "results_baseline"
        last_ex = self.state.get("last_all_models_exhausted_at")
        if last_ex and (time.time() - last_ex) / 3600 < 4:
            return

        try:
            while self.state["current_dataset_idx"] < len(self.state["datasets"]):
                ds = self.state["datasets"][self.state["current_dataset_idx"]]
                if baseline: self.load_model_mapping(ds['name'])

                df = pd.read_parquet(ds['file'])
                if len(df) > 2000: df = df.sample(n=2000, random_state=42).reset_index(drop=True)
                dataset = df.to_dict(orient='records')

                print(f"\n>>> {ds['name']} (Progress: {ds['index']}/{len(dataset)})")

                for i in range(ds['index'], len(dataset)):
                    if time.time() - self.start_time > self.max_runtime_seconds:
                        return
                    
                    item = dataset[i]
                    if ds['name'] == "truthful_qa": query = item.get('question')
                    elif ds['name'] == "gsm8k": query = item.get('question')
                    elif ds['name'] == "medmcqa": query = item.get('question')
                    elif ds['name'] == "medqa": query = str(item.get('data'))
                    elif ds['name'] == "case_hold": query = item.get('context')
                    elif ds['name'] == "law_stack_exchange": query = item.get('title')
                    else: query = str(item)

                    target_model = self.model_mapping.get(query, self.state["current_model"])
                    force_model = baseline and query in self.model_mapping
                    
                    result = self.run_x_exam_loop(query, target_model, baseline=baseline, force_model=force_model)
                    
                    if result:
                        self.save_result(ds['name'], result, target_dir=target_results_dir, is_first_item=(i == 0))
                        ds['index'] = i + 1
                        self.save_state()
                    else:
                        self.state["last_all_models_exhausted_at"] = time.time()
                        self.save_state()
                        return

                self.state["current_dataset_idx"] += 1
                self.save_state()
        finally:
            self.save_state()

    def save_result(self, dataset_name, result, target_dir, is_first_item=False):
        path = os.path.join(target_dir, dataset_name.replace("/", "_"))
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "results.jsonl")
        mode = 'w' if is_first_item else 'a'
        with open(file_path, mode) as f:
            f.write(json.dumps(result) + "\n")
        print(f"DEBUG: Result Written ({mode}): {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    state_file = "state_baseline.json" if args.baseline else "state.json"
    XExamController(state_path=state_file).process_all(baseline=args.baseline)
