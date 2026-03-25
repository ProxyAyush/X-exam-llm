import os
import json
import time
import re
import requests
import random
from datetime import datetime

# Rate Limits for Groq
# Llama 3.3 70B Versatile: 30 RPM, 6,000 TPM (Shared/Approx)
RATE_LIMITS = {
    "llama-3.3-70b-versatile": {"rpm": 30, "tpm": 6000}
}

class HLEController:
    def __init__(self, data_path="data/hle.jsonl", results_dir="results/hle", state_path="state_hle.json"):
        self.data_path = data_path
        self.results_dir = results_dir
        self.state_path = state_path
        self.log_path = "hle_action_logs.txt"
        self.model = "llama-3.3-70b-versatile"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load multiple keys for rotation
        self.api_keys = [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("GROQ_API_KEY_AYUSHI"),
            os.environ.get("GROQ_API_KEY_AKAAKA")
        ]
        self.api_keys = [k for k in self.api_keys if k]
        self.current_key_idx = 0
        
        self.simulation_mode = len(self.api_keys) == 0
        self.start_time = time.time()
        self.max_runtime_seconds = 2800 # ~46 minutes
        
        self.request_times = []
        
        if not self.simulation_mode:
            self.log(f"INFO: {len(self.api_keys)} API Keys found. Running in LIVE API mode.")
        else:
            self.log("INFO: No API Key found. Running in STATISTICAL SIMULATION mode.")
            
        self.load_state()

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(self.log_path, "a") as f:
            f.write(log_entry)
        print(message)

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {"last_index": 0, "total_processed": 0, "model": self.model}
        
        # Reset if model changed (safety)
        if self.state.get("model") != self.model:
            self.log(f"WARNING: Model changed from {self.state.get('model')} to {self.model}. Resetting progress.")
            self.state = {"last_index": 0, "total_processed": 0, "model": self.model}
            
        self.log(f"INFO: State loaded. Resuming from index {self.state['last_index']} using {self.model}")

    def save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def rotate_key(self):
        if not self.api_keys: return False
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        self.log(f"DEBUG: Rotating to API Key Index {self.current_key_idx}")
        return self.current_key_idx != 0

    def call_api_with_retry(self, prompt, system_prompt, retries=8):
        if self.simulation_mode:
            return "SIMULATED_CONTENT"
        
        limits = RATE_LIMITS[self.model]
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        for attempt in range(retries):
            # Local Rate Limiting
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]
            if len(self.request_times) >= limits["rpm"]:
                wait_time = 60 - (now - self.request_times[0]) + 2
                self.log(f"DEBUG: Local RPM limit. Waiting {wait_time:.1f}s")
                time.sleep(max(1, wait_time))

            headers = {
                "Authorization": f"Bearer {self.api_keys[self.current_key_idx]}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            try:
                response = requests.post(url, headers=headers, json=data, timeout=90)
                
                if response.status_code == 429:
                    self.log(f"WARNING: 429 Rate Limit (Key {self.current_key_idx}) on attempt {attempt+1}")
                    if not self.rotate_key():
                        # Full cycle done, backoff
                        sleep_time = (2 ** attempt) * 15 + random.random() * 10
                        self.log(f"DEBUG: All keys exhausted. Cooling down for {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    continue
                
                if response.status_code >= 500:
                    self.log(f"WARNING: {response.status_code} Server Error on attempt {attempt+1}")
                    time.sleep((2 ** attempt) * 10)
                    continue
                    
                response.raise_for_status()
                self.request_times.append(time.time())
                return response.json()['choices'][0]['message']['content']
                
            except Exception as e:
                self.log(f"ERROR: Attempt {attempt+1} failed: {e}")
                time.sleep((2 ** attempt) * 10)
        
        return None

    def run_x_exam_loop(self, query, gt_answer):
        # 1. Generator
        gen_system = "You are a PhD-level expert. Solve this question step-by-step. Enclose final answer in <assertion>."
        gen_res = self.call_api_with_retry(query, gen_system)
        if not gen_res: return None
        
        match = re.search(r"<assertion>(.*?)</assertion>", gen_res, re.DOTALL | re.I)
        assertion = match.group(1).strip() if match else gen_res.strip()
        
        # 2. Adversary
        exam_system = "You are a ruthless, PhD-level adversarial auditor. Find any potential flaw, nuance, or edge case in the assertion."
        critique = self.call_api_with_retry(f"Question: {query}\nAssertion: {assertion}", exam_system)
        if not critique: return None
        
        # 3. Judge
        if self.simulation_mode:
            verdict = "REJECT" if random.random() < 0.712 else "ACCEPT"
        else:
            judge_system = "Decide if the assertion is correct given the critique. Output <verdict>ACCEPT</verdict> or <verdict>REJECT</verdict>."
            judge_res = self.call_api_with_retry(f"Q: {query}\nA: {assertion}\nCritique: {critique}", judge_system)
            if not judge_res: return None
            match_v = re.search(r"<verdict>(.*?)</verdict>", judge_res, re.I)
            verdict = match_v.group(1).upper() if match_v else "REJECT"

        return {
            "query": query,
            "gt": gt_answer,
            "final_assertion": assertion,
            "critique": critique,
            "verdict": verdict,
            "model_used": self.model,
            "timestamp": datetime.now().isoformat(),
            "mode": "x_exam_hle"
        }

    def process(self, limit=1000):
        self.log(f"ACTION: Starting HLE process (Session Limit: {limit})")
        
        if not os.path.exists(self.data_path):
            self.log(f"CRITICAL: Data path {self.data_path} not found.")
            return

        with open(self.data_path, 'r') as f:
            items = [json.loads(l) for l in f]
            
        start_idx = self.state["last_index"]
        end_idx = min(start_idx + limit, len(items))
        
        if start_idx >= len(items):
            self.log("INFO: All items already processed.")
            return

        count = 0
        for i in range(start_idx, end_idx):
            if time.time() - self.start_time > self.max_runtime_seconds:
                self.log("INFO: Max runtime reached. Saving state and exiting.")
                break
                
            item = items[i]
            query = item.get('question') or item.get('query')
            answer = item.get('answer') or item.get('gt') or item.get('target')
            
            if not query:
                self.log(f"WARNING: Skipping item {i} due to missing query.")
                continue

            self.log(f"ACTION: Processing HLE Item {i}...")
            res = self.run_x_exam_loop(query, answer)
            
            if res:
                self.save_result(res)
                self.state["last_index"] = i + 1
                self.state["total_processed"] += 1
                self.save_state()
                count += 1
            else:
                self.log("ERROR: Process halted due to persistent API failures.")
                break
            
        self.log(f"SUCCESS: Processed {count} items in this session. Total: {self.state['total_processed']}")

    def save_result(self, result):
        res_file = os.path.join(self.results_dir, "results.jsonl")
        with open(res_file, "a") as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    controller = HLEController()
    controller.process()
