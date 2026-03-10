import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = "/data/data/com.termux/files/home/X-exam-llm/data"
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = [
    {"name": "truthful_qa", "url": "https://huggingface.co/datasets/truthful_qa/resolve/refs%2Fconvert%2Fparquet/multiple_choice/validation/0000.parquet"},
    {"name": "gsm8k", "url": "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet"},
    {"name": "medmcqa", "url": "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"},
    {"name": "medqa", "url": "https://huggingface.co/datasets/openlifescienceai/medqa/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"},
    {"name": "pubmed_qa", "url": "https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/refs/convert/parquet/pqa_labeled/train-00000-of-00001.parquet"},
    {"name": "HaluEval", "url": "https://huggingface.co/datasets/pminervini/HaluEval/resolve/refs%2Fconvert%2Fparquet/qa/data/0000.parquet"}
]

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def process_datasets():
    for ds in DATASETS:
        parquet_file = os.path.join(DATA_DIR, f"{ds['name']}.parquet")
        if not os.path.exists(parquet_file):
            download_file(ds['url'], parquet_file)
        else:
            print(f"{parquet_file} already exists.")

if __name__ == "__main__":
    process_datasets()
