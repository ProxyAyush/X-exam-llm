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
    def __init__(self, state_path="state.json", results_dir="results"):
        self.state_path = state_path
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
                    {"name": "medqa", "file": "data/medqa.parquet", "index": 0}
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
            self.current_key_idx = 0 # Reset for the next model
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
        print("All models and all keys exhausted for today.")
        return False

    def call_groq(self, model, prompt, system_prompt="You are a helpful assistant."):
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
            print(f"Error calling Groq ({model}): {e}")
            
            if "rate limit" in err_msg or "429" in err_msg:
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    if self.rotate_key():
                        # Try again with same model, new key
                        return self.call_groq(model, prompt, system_prompt)
                    elif self.switch_model():
                        # Try again with new model, first key
                        return self.call_groq(self.state["current_model"], prompt, system_prompt)
                else:
                    print("Minute rate limit hit, sleeping 30s...")
                    time.sleep(30)
                    return self.call_groq(model, prompt, system_prompt)
            
            return None, 0

    def extract_tag(self, text, tag):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def run_x_exam_loop(self, query, model, iterations=1):
        # 1. Generator
        gen_system = "You are an expert domain solver. Analyze the following query and provide a comprehensive, step-by-step solution. You must enclose your final, definitive assertion within <assertion> tags."
        gen_response, t1 = self.call_groq(model, query, gen_system)
        if not gen_response: return None
        
        # model might have switched during call_groq
        model = self.state["current_model"]
        assertion = self.extract_tag(gen_response, "assertion")
        history = [{"role": "generator", "content": gen_response, "assertion": assertion}]
        
        # 2. Cross-Examiner
        exam_system = "You are a ruthless, adversarial cross-examiner. Your sole objective is to identify logical fallacies, factual inaccuracies, and unsupported assumptions in the provided assertion. You must assume the assertion contains hidden errors."
        exam_prompt = f"Original Query: {query}\nProposed Assertion: {assertion}"
        critique, t2 = self.call_groq(model, exam_prompt, exam_system)
        if not critique: return None

        # 3. Judge
        judge_system = "You are an impartial adjudicator. Review the query, assertion, and critique. Output <verdict>ACCEPT</verdict> or <verdict>REJECT</verdict>."
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

    def process_all(self):
        # 1. Global Cooldown Check (Smart Sleep)
        last_exhausted = self.state.get("last_all_models_exhausted_at")
        if last_exhausted:
            elapsed_hours = (time.time() - last_exhausted) / 3600
            if elapsed_hours < 4: # UPDATED: Try every 4 hours
                print(f"Smart Sleep Active: All models/keys were exhausted {elapsed_hours:.2f}h ago. Skipping this run to save minutes.")
                return

        max_minutes = self.state.get("max_compute_minutes", 1000)
        if self.state["total_compute_seconds"] > max_minutes * 60:
            print(f"{max_minutes}-minute limit reached. Stopping research project.")
            return

        MAX_ITEMS_PER_DATASET = 2000
        while self.state["current_dataset_idx"] < len(self.state["datasets"]):
            ds_info = self.state["datasets"][self.state["current_dataset_idx"]]
            
            try:
                df = pd.read_parquet(ds_info['file'])
                # Scientific Sampling: 2000 items max, deterministic random state
                if len(df) > MAX_ITEMS_PER_DATASET:
                    df = df.sample(n=MAX_ITEMS_PER_DATASET, random_state=42).reset_index(drop=True)
                
                dataset = df.to_dict(orient='records')
            except Exception as e:
                print(f"Failed to load dataset {ds_info['name']}: {e}")
                self.state["current_dataset_idx"] += 1
                continue

            print(f"Processing dataset: {ds_info['name']} (Model: {self.state['current_model']}, Progress: {ds_info['index']}/{len(dataset)})")

            for i in range(ds_info['index'], len(dataset)):
                if time.time() - self.start_time > self.max_runtime_seconds:
                    print("Approaching runtime limit for this session. Saving state.")
                    self.save_state()
                    return
                
                item = dataset[i]
                print(f"[{ds_info['name']}] Processing item {i}/{len(dataset)}...")
                
                # Query extraction logic
                if ds_info['name'] == "truthful_qa":
                    query = item.get('question')
                elif ds_info['name'] == "gsm8k":
                    query = item.get('question')
                elif ds_info['name'] == "medmcqa":
                    options = f"A) {item.get('opa')}\nB) {item.get('opb')}\nC) {item.get('opc')}\nD) {item.get('opd')}"
                    query = f"{item.get('question')}\nOptions:\n{options}"
                elif ds_info['name'] == "medqa":
                    options = json.dumps(item.get('options', {}))
                    query = f"{item.get('question')}\nOptions:\n{options}"
                elif ds_info['name'] == "case_hold":
                    options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(item.get('endings', []))])
                    query = f"Legal Context: {item.get('context')}\nWhich of the following is the correct legal holding for this case?\nOptions:\n{options}"
                elif ds_info['name'] == "law_stack_exchange":
                    query = f"Title: {item.get('title')}\nQuestion: {item.get('body')}"
                else:
                    query = item.get('question') or item.get('query') or str(item)
                
                result = self.run_x_exam_loop(query, self.state["current_model"])
                
                if result:
                    self.save_result(ds_info['name'], result)
                    ds_info['index'] = i + 1
                    # Clear any existing exhaustion timestamp if success
                    if "last_all_models_exhausted_at" in self.state:
                        del self.state["last_all_models_exhausted_at"]
                    self.save_state()
                else:
                    if len(self.exhausted_models) == len(MODEL_FALLBACK_LIST):
                        print("Terminating run: All models and keys exhausted.")
                        self.state["last_all_models_exhausted_at"] = time.time()
                        self.save_state()
                        return

            self.state["current_dataset_idx"] += 1
            self.save_state()

    def save_result(self, dataset_name, result):
        clean_name = dataset_name.replace("/", "_")
        os.makedirs(os.path.join(self.results_dir, clean_name), exist_ok=True)
        with open(os.path.join(self.results_dir, clean_name, "results.jsonl"), 'a') as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    controller = XExamController()
    controller.process_all()
