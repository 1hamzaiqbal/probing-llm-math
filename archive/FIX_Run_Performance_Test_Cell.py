# --- FIXED: Run Performance Test (simple generation + heatmap) ---
import re, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# You can toggle actual generation off to speed up/debug the rest:
USE_MODEL = True  # set to False to skip generation if needed

# 1) Make a small, stratified sample
SEED = 42
SAMPLE_SIZE = max(15, min(60, len(df)))  # pick something sensible by default
group_cols = [c for c in ["topic","difficulty"] if c in df.columns]
if group_cols:
    gb = df.groupby(group_cols, dropna=False)
    k = max(1, SAMPLE_SIZE // max(1, len(gb)))
    parts = []
    for _, g in gb:
        parts.append(g.sample(n=min(k, len(g)), random_state=SEED))
    eval_df = pd.concat(parts, ignore_index=True).sample(n=min(SAMPLE_SIZE, sum(map(len, parts))), random_state=SEED)
else:
    eval_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).copy()

# 2) Ensure model_answer column exists  
if "model_answer" not in eval_df.columns:
    eval_df["model_answer"] = ""

# 3) Minimal model generation (if model/tokenizer exist and USE_MODEL)
def have_model():
    return ("model" in globals()) and ("tokenizer" in globals())

if USE_MODEL and have_model():
    model.eval()
    device = next(model.parameters()).device

    def generate_answer(q: str) -> str:
        prompt = f"Problem:\n{q}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # take only the part after 'Answer:'
        ans = text.split("Answer:", 1)[-1].strip()
        return ans

    # Generate only for rows with empty model_answer
    need = eval_df["model_answer"].isna() | (eval_df["model_answer"].astype(str).str.strip() == "")
    if need.any():
        print(f"Generating answers for {need.sum()} problems ...")
        eval_df.loc[need, "model_answer"] = eval_df.loc[need, "question"].map(generate_answer)
else:
    print("Skipping generation (either USE_MODEL=False or model/tokenizer not loaded).")

# 4) Compute correctness via lenient substring rule (helpers from the added cell)
eval_df = compute_is_correct(eval_df)

# 5) Basic analysis + heatmap
overall = float(eval_df["is_correct"].mean()) if len(eval_df) else float("nan")
print(f"Overall accuracy (substring rule): {overall:.3f} on n={len(eval_df)}")

# Accuracy by topic
if "topic" in eval_df.columns:
    acc_by_topic = eval_df.groupby("topic")["is_correct"].mean().sort_values(ascending=False)
    print("\nAccuracy by topic:\n", acc_by_topic)

# Heatmap: topic x difficulty
if {"topic","difficulty"}.issubset(eval_df.columns):
    def diff_key(x):
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else 10**9
    topics = sorted([str(t) for t in eval_df["topic"].dropna().unique()])
    diffs = sorted([str(d) for d in eval_df["difficulty"].dropna().unique()], key=diff_key)
    pivot = eval_df.pivot_table(index="topic", columns="difficulty", values="is_correct", aggfunc="mean") \
                   .reindex(index=topics, columns=diffs)
    plt.figure(figsize=(max(6, 1 + 0.8*len(diffs)), max(5, 0.5*len(topics))))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(diffs)), diffs, rotation=45, ha="right")
    plt.yticks(range(len(topics)), topics)
    plt.title("Accuracy Heatmap (Topic × Difficulty)")
    plt.xlabel("Difficulty")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.show()
else:
    print("Heatmap skipped (need both 'topic' and 'difficulty' columns).")

# Save results
eval_df.to_csv("performance_test_results.csv", index=False)
print("Saved: performance_test_results.csv")