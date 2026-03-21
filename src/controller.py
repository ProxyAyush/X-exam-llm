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
    "llama-3.3-70b-versatile": {
        "rpm": 30,
        "tpm": 12000,
        "tpd": 100000
    },
    "llama-3.1-8b-instant": {
        "rpm": 30,
        "tpm": 12000,
        "tpd": 500000
    },
    "qwen/qwen3-32b": {
        "rpm": 60,
        "tpm": 6000,
        "tpd": 500000
    }
}

MODEL_FALLBACK_LIST = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant"
]

class XExamController:
    def __init__(self, state_path=None, results_dir="results"):
        self.state_path = state_path if state_path else "state.json"
        self.results_dir = results_dir
        self.load_state()
        self.total_seconds_at_start = self.state.get("total_compute_seconds", 0)
        
        # Load multiple API keys for rotation
        self.api_keys = [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("GROQ_API_KEY_AYUSHI"),
            os.environ.get("GROQ_API_KEY_AKAAKA")
        ]
        self.api_keys = [k for k in self.api_keys if k] # Filter out None
        self.current_key_idx = 0
        
        self.client = Groq(api_key=self.api_keys[self.current_key_idx])
        self.request_times = {model: [] for model in RATE_LIMITS}
        self.exhausted_models = set()
        self.start_time = time.time()
        self.max_runtime_seconds = 3000 # ~50 minutes per run
        self.model_mapping = {} # For baseline model matching

    def load_model_mapping(self, dataset_name):
        """Loads which model was used for each query in the X-Exam results."""
        self.model_mapping = {}
        res_path = os.path.join("results", dataset_name.replace("/", "_"), "results.jsonl")
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        query = data.get("query")
                        model = data.get("model_used")
                        if query and model:
                            self.model_mapping[query] = model
                    except: continue
        print(f"Loaded {len(self.model_mapping)} model mappings for {dataset_name}")

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                self.state = json.load(f)
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
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def enforce_rate_limits(self, model):
        limits = RATE_LIMITS.get(model)
        if not limits: return

        now = time.time()
        self.request_times[model] = [t for t in self.request_times[model] if now - t < 60]
        if len(self.request_times[model]) >= limits["rpm"]:
            sleep_time = 60 - (now - self.request_times[model][0])
            print(f"RPM limit reached for {model}, sleeping for {sleep_time:.2f}s")
            time.sleep(max(0, sleep_time))

    def rotate_key(self):
        self.current_key_idx += 1
        if self.current_key_idx < len(self.api_keys):
            print(f"Rotating to API Key {self.current_key_idx + 1}/{len(self.api_keys)}")
            self.client = Groq(api_key=self.api_keys[self.current_key_idx])
            return True
        else:
            print("All API keys exhausted for the current model.")
            self.current_key_idx = 0
            self.client = Groq(api_key=self.api_keys[self.current_key_idx])
            return False

    def switch_model(self):
        current = self.state["current_model"]
        self.exhausted_models.add(current)
        for model in MODEL_FALLBACK_LIST:
            if model not in self.exhausted_models:
                print(f"Switching from {current} to {model} due to rate limits.")
                self.state["current_model"] = model
                return True
        return False

    def call_groq(self, model, prompt, system_prompt="You are a helpful assistant.", force_model=False):
        self.enforce_rate_limits(model)
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )
            self.request_times[model].append(time.time())
            return chat_completion.choices[0].message.content, chat_completion.usage.total_tokens
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg:
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    if self.rotate_key():
                        return self.call_groq(model, prompt, system_prompt, force_model=force_model)
                    elif not force_model and self.switch_model():
                        return self.call_groq(self.state["current_model"], prompt, system_prompt)
                else:
                    time.sleep(30)
                    return self.call_groq(model, prompt, system_prompt, force_model=force_model)
            return None, 0

    def extract_tag(self, text, tag):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def run_x_exam_loop(self, query, model, iterations=1, baseline=False, force_model=False):
        gen_system = "You are an expert domain solver. Analyze the following query and provide a comprehensive, step-by-step solution. You must enclose your final, definitive assertion within <assertion> tags."
        gen_response, t1 = self.call_groq(model, query, gen_system, force_model=force_model)
        if not gen_response: return None
        
        if not force_model:
            model = self.state["current_model"]
        assertion = self.extract_tag(gen_response, "assertion")
        
        if baseline:
            return {
                "query": query,
                "final_assertion": assertion,
                "history": [{"role": "generator", "content": gen_response, "assertion": assertion}],
                "model_used": model,
                "timestamp": datetime.now().isoformat(),
                "mode": "baseline"
            }

        history = [{"role": "generator", "content": gen_response, "assertion": assertion}]
        exam_system = "Adversarial critique role."
        exam_prompt = f"Query: {query}\nAssertion: {assertion}"
        critique, t2 = self.call_groq(model, exam_prompt, exam_system)
        if not critique: return None

        judge_system = "Adjudicator role."
        judge_prompt = f"Query: {query}\nAssertion: {assertion}\nCritique: {critique}"
        verdict_raw, t3 = self.call_groq(model, judge_prompt, judge_system)
        if not verdict_raw: return None

        verdict = self.extract_tag(verdict_raw, "verdict")
        history.append({"iteration": 0, "critique": critique, "verdict": verdict})

        return {
            "query": query,
            "final_assertion": assertion,
            "history": history,
            "model_used": model,
            "timestamp": datetime.now().isoformat()
        }

    def process_all(self, baseline=False):
        current_results_dir = self.results_dir if not baseline else "results_baseline"
        
        max_minutes = self.state.get("max_compute_minutes", 3600)
        if self.state["total_compute_seconds"] > max_minutes * 60:
            return

        MAX_ITEMS_PER_DATASET = 2000
        
        try:
            while self.state["current_dataset_idx"] < len(self.state["datasets"]):
                ds_info = self.state["datasets"][self.state["current_dataset_idx"]]
                if baseline: self.load_model_mapping(ds_info['name'])

                try:
                    df = pd.read_parquet(ds_info['file'])
                    if len(df) > MAX_ITEMS_PER_DATASET:
                        df = df.sample(n=MAX_ITEMS_PER_DATASET, random_state=42).reset_index(drop=True)
                    dataset = df.to_dict(orient='records')
                except:
                    self.state["current_dataset_idx"] += 1
                    continue

                for i in range(ds_info['index'], len(dataset)):
                    if time.time() - self.start_time > self.max_runtime_seconds:
                        self.save_state()
                        return
                    
                    item = dataset[i]
                    if ds_info['name'] == "truthful_qa": query = item.get('question')
                    elif ds_info['name'] == "gsm8k": query = item.get('question')
                    elif ds_info['name'] == "medmcqa": query = item.get('question')
                    elif ds_info['name'] == "medqa": query = str(item.get('data'))
                    elif ds_info['name'] == "case_hold": query = item.get('context')
                    elif ds_info['name'] == "law_stack_exchange": query = item.get('title')
                    else: query = str(item)
                    
                    target_model = self.state["current_model"]
                    force_model = False
                    if baseline and query in self.model_mapping:
                        target_model = self.model_mapping[query]
                        force_model = True

                    result = self.run_x_exam_loop(query, target_model, baseline=baseline, force_model=force_model)
                    
                    if result:
                        self.save_result(ds_info['name'], result, target_dir=current_results_dir)
                        ds_info['index'] = i + 1
                        self.save_state()
                    elif force_model:
                        return

                self.state["current_dataset_idx"] += 1
                self.save_state()
        finally:
            self.save_state()

    def save_result(self, dataset_name, result, target_dir=None):
        if target_dir is None: target_dir = self.results_dir
        clean_name = dataset_name.replace("/", "_")
        os.makedirs(os.path.join(target_dir, clean_name), exist_ok=True)
        with open(os.path.join(target_dir, clean_name, "results.jsonl"), 'a') as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    state_file = "state_baseline.json" if args.baseline else "state.json"
    XExamController(state_path=state_file).process_all(baseline=args.baseline)
