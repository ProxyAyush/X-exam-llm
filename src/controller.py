import os
import json
import time
import argparse
import re
import pandas as pd
from datetime import datetime
from groq import Groq
from tqdm import tqdm

# Rate limit configurations (as of March 2026)
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
            self.state = {"current_dataset_idx": 0, "current_model": "llama-3.3-70b-versatile", "total_compute_seconds": 0, "datasets": []}

    def save_state(self):
        self.state["total_compute_seconds"] = self.total_seconds_at_start + (time.time() - self.start_time)
        with open(self.state_path, 'w') as f: json.dump(self.state, f, indent=2)
        print(f"DEBUG: Progress State Saved: {self.state_path}")

    def call_groq(self, model, prompt, system_prompt="You are a helpful assistant.", force_model=False):
        # Rate limit enforcement
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
            return chat_completion.choices[0].message.content, chat_completion.usage.total_tokens
        except Exception as e:
            err_msg = str(e).lower()
            print(f"API ERROR ({model}): {e}")
            if "rate limit" in err_msg or "429" in err_msg:
                # Rotate keys for TPD limits
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    self.current_key_idx += 1
                    if self.current_key_idx < len(self.api_keys):
                        self.client = Groq(api_key=self.api_keys[self.current_key_idx])
                        return self.call_groq(model, prompt, system_prompt, force_model=force_model)
                    elif not force_model:
                        # Only switch model if not forced
                        self.exhausted_models.add(model)
                        for m in MODEL_FALLBACK_LIST:
                            if m not in self.exhausted_models:
                                self.state["current_model"] = m
                                return self.call_groq(m, prompt, system_prompt)
            return None, 0

    def run_x_exam_loop(self, query, model, iterations=1, baseline=False, force_model=False):
        gen_system = "You are an expert domain solver. Provide a step-by-step solution. You must enclose your final assertion within <assertion> tags."
        gen_response, tokens = self.call_groq(model, query, gen_system, force_model=force_model)
        if not gen_response: 
            print(f"DEBUG: API Failed to return response for model {model}")
            return None
        
        match = re.search(f"<assertion>(.*?)</assertion>", gen_response, re.DOTALL | re.IGNORECASE)
        assertion = match.group(1).strip() if match else gen_response.strip()
        
        print(f"DEBUG: Success - Generated Assertion (Length: {len(assertion)}) using Model: {model}")
        
        return {
            "query": query,
            "final_assertion": assertion,
            "history": [{"role": "generator", "content": gen_response, "assertion": assertion}],
            "model_used": model,
            "timestamp": datetime.now().isoformat(),
            "mode": "baseline" if baseline else "x_exam"
        }

    def process_all(self, baseline=False):
        results_dir = self.results_dir if not baseline else "results_baseline"
        
        try:
            while self.state["current_dataset_idx"] < len(self.state["datasets"]):
                ds = self.state["datasets"][self.state["current_dataset_idx"]]
                if baseline: self.load_model_mapping(ds['name'])

                df = pd.read_parquet(ds['file'])
                if len(df) > 2000: df = df.sample(n=2000, random_state=42).reset_index(drop=True)
                dataset = df.to_dict(orient='records')

                print(f"\n>>> PROCESSING DATASET: {ds['name']} (Starting at Index: {ds['index']})")

                for i in range(ds['index'], len(dataset)):
                    if time.time() - self.start_time > 3000: 
                        print("DEBUG: Approaching 50m runtime. Terminating session.")
                        return
                    
                    item = dataset[i]
                    if ds['name'] == "truthful_qa": query = item.get('question')
                    elif ds_info['name'] == "gsm8k": query = item.get('question')
                    elif ds_info['name'] == "medmcqa": query = item.get('question')
                    elif ds_info['name'] == "medqa": query = str(item.get('data'))
                    elif ds_info['name'] == "case_hold": query = item.get('context')
                    elif ds_info['name'] == "law_stack_exchange": query = item.get('title')
                    else: query = str(item)

                    target_model = self.model_mapping.get(query, self.state["current_model"])
                    force_model = baseline and query in self.model_mapping
                    
                    if force_model: print(f"DEBUG: [ITEM {i}] Forcing Phase 3 Model: {target_model}")

                    result = self.run_x_exam_loop(query, target_model, baseline=baseline, force_model=force_model)
                    
                    if result:
                        self.save_result(ds['name'], result, target_dir=results_dir)
                        ds['index'] = i + 1
                        self.save_state()
                    elif force_model:
                        print(f"CRITICAL: Baseline Model {target_model} exhausted. Stopping for 4-hour sleep.")
                        return

                self.state["current_dataset_idx"] += 1
                self.save_state()
        finally:
            self.save_state()

    def save_result(self, dataset_name, result, target_dir):
        path = os.path.join(target_dir, dataset_name.replace("/", "_"))
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "results.jsonl")
        with open(file_path, 'a') as f:
            f.write(json.dumps(result) + "\n")
        print(f"DEBUG: Result Written to Disk: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    state_file = "state_baseline.json" if args.baseline else "state.json"
    XExamController(state_path=state_file).process_all(baseline=args.baseline)
