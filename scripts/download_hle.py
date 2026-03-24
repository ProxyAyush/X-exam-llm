import pandas as pd
try:
    from datasets import load_dataset
    print("Loading HLE dataset from Hugging Face...")
    ds = load_dataset("cais/hle", split="test")
    df = ds.to_pandas()
    df.to_parquet("data/hle.parquet")
    print(f"HLE dataset saved to data/hle.parquet. Total items: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading HLE: {e}")
