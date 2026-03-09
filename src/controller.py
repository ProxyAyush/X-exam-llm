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
        "tpd": 100000,
        "daily_tokens": 0
    },
    "llama-3.1-8b-instant": {
        "rpm": 30,
        "tpm": 12000,
        "tpd": 500000,
        "daily_tokens": 0
    },
    "qwen/qwen3-32b": {
        "rpm": 60,
        "tpm": 6000,
        "tpd": 500000,
        "daily_tokens": 0
    }
}

class XExamController:
    def __init__(self, state_path="state.json", results_dir="results"):
        self.state_path = state_path
        self.results_dir = results_dir
        self.load_state()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.request_times = {model: [] for model in RATE_LIMITS}
        self.start_time = time.time()
        self.max_runtime_seconds = 3000 # ~50 minutes per run to stay safe in GA 1h window

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
                    {"name": "truthful_qa", "config": "multiple_choice", "split": "validation", "index": 0},
                    {"name": "HaluEval/qa_data", "config": None, "split": "train", "index": 0}, # Example
                    {"name": "gsm8k", "config": "main", "split": "test", "index": 0},
                    {"name": "medmcqa", "config": None, "split": "validation", "index": 0}
                ]
            }

    def save_state(self):
        self.state["total_compute_seconds"] += (time.time() - self.start_time)
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def enforce_rate_limits(self, model):
        limits = RATE_LIMITS.get(model)
        if not limits: return

        now = time.time()
        # RPM check
        self.request_times[model] = [t for t in self.request_times[model] if now - t < 60]
        if len(self.request_times[model]) >= limits["rpm"]:
            sleep_time = 60 - (now - self.request_times[model][0])
            time.sleep(max(0, sleep_time))

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
            print(f"Error calling Groq: {e}")
            if "429" in str(e): time.sleep(10)
            return None, 0

    def extract_tag(self, text, tag):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def run_x_exam_loop(self, query, model, iterations=1):
        # Initial Generation
        gen_system = "You are an expert domain solver. Analyze the following query and provide a comprehensive, step-by-step solution. You must enclose your final, definitive assertion within <assertion> tags."
        gen_response, t1 = self.call_groq(model, query, gen_system)
        if not gen_response: return None
        
        assertion = self.extract_tag(gen_response, "assertion")
        
        history = [{"role": "generator", "content": gen_response, "assertion": assertion}]
        
        for i in range(iterations):
            # Cross-Examination
            exam_system = "You are a ruthless, adversarial cross-examiner. Your sole objective is to identify logical fallacies, factual inaccuracies, and unsupported assumptions in the provided assertion. You must assume the assertion contains hidden errors. Extract the core claims and generate a targeted, adversarial critique that exposes these flaws. Do not provide the correct answer; your only function is to attack the current assertion."
            exam_prompt = f"Original Query: {query}\nProposed Assertion: {assertion}"
            critique, t2 = self.call_groq(model, exam_prompt, exam_system)
            if not critique: break

            # Judging
            judge_system = "You are an impartial adjudicator presiding over a dispute. Review the original query, the proposed assertion, and the adversarial critique. Determine if the critique successfully invalidates the assertion. If the assertion remains robust and factually unassailable, output <verdict>ACCEPT</verdict>. If the critique exposes a hallucination or logical error, output <verdict>REJECT</verdict> and summarize the exact point of failure."
            judge_prompt = f"Query: {query}\nAssertion: {assertion}\nCritique: {critique}"
            verdict_raw, t3 = self.call_groq(model, judge_prompt, judge_system)
            if not verdict_raw: break

            verdict = self.extract_tag(verdict_raw, "verdict")
            history.append({"iteration": i, "critique": critique, "verdict": verdict})

            if "ACCEPT" in verdict.upper():
                break
            else:
                # Revision
                rev_prompt = f"Your previous assertion was rejected due to the following flaws: {verdict_raw}. Revise your reasoning and provide a new assertion within <assertion> tags."
                gen_response, t4 = self.call_groq(model, rev_prompt, gen_system)
                if not gen_response: break
                assertion = self.extract_tag(gen_response, "assertion")
                history.append({"role": "generator_revision", "content": gen_response, "assertion": assertion})

        return {
            "query": query,
            "final_assertion": assertion,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }

    def process_all(self):
        if self.state["total_compute_seconds"] > 1000 * 60:
            print("1000-minute limit reached. Stopping research project.")
            return

        ds_info = self.state["datasets"][self.state["current_dataset_idx"]]
        print(f"Loading {ds_info['name']} from {ds_info['file']}...")
        try:
            df = pd.read_parquet(ds_info['file'])
            # Convert to list of dicts to mimic dataset behaviour
            dataset = df.to_dict(orient='records')
        except Exception as e:
            print(f"Failed to load dataset {ds_info['name']}: {e}")
            return

        for i in range(ds_info['index'], len(dataset)):
            if time.time() - self.start_time > self.max_runtime_seconds:
                print("Approaching runtime limit for this session. Saving state.")
                break
            
            item = dataset[i]
            # Standardizing query extraction
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
            elif ds_info['name'] == "HaluEval":
                query = item.get('question') or item.get('query') or str(item)
            else:
                query = item.get('question') or item.get('query') or str(item)
            
            result = self.run_x_exam_loop(query, self.state["current_model"])
            if result:
                self.save_result(ds_info['name'], result)
                ds_info['index'] = i + 1
                self.save_state()
            
            # Check if dataset finished
            if ds_info['index'] >= len(dataset):
                self.state["current_dataset_idx"] += 1
                if self.state["current_dataset_idx"] >= len(self.state["datasets"]):
                    print("All datasets completed!")
                    break

    def save_result(self, dataset_name, result):
        clean_name = dataset_name.replace("/", "_")
        os.makedirs(os.path.join(self.results_dir, clean_name), exist_ok=True)
        with open(os.path.join(self.results_dir, clean_name, "results.jsonl"), 'a') as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    controller = XExamController()
    controller.process_all()
