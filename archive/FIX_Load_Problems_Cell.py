# --- FIXED: Load Problems (robust path resolution) ---
import os, pandas as pd

def find_problems_csv():
    candidates = [
        "probing-llm-math/data/problems.csv",
        "./probing-llm-math/data/problems.csv",
        "data/problems.csv",
        "./data/problems.csv",
        "/content/probing-llm-math/data/problems.csv",   # Colab fallback
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find problems.csv. Make sure the repo was cloned to ./probing-llm-math")

csv_path = find_problems_csv()
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} problems from: {csv_path}")
print("Columns:", list(df.columns))
print("Sample:\n", df.head(2)[["id","topic","difficulty","question","answer"]])