# Filtered Probe Analysis - Colab Cells

Run these cells in Colab to train probes **without** the 1000+ token samples.

---

## Cell 1: Setup & Pull Latest Code

```python
# Pull latest code with filtered training script
!cd probing-llm-math && git pull

# Verify the new script exists
!ls -la probing-llm-math/src/train_probe_filtered.py
```

---

## Cell 2: Check Current Data Files

```python
import os

# Check what data files exist
print("Current directory files:")
for f in os.listdir('.'):
    if f.endswith('.pt') or f.endswith('.csv'):
        print(f"  {f}")

print("\nDrive files (if mounted):")
drive_path = "/content/drive/MyDrive/probe_results"
if os.path.exists(drive_path):
    for f in os.listdir(drive_path):
        print(f"  {f}")
```

---

## Cell 3: Analyze Token Distribution BEFORE Filtering

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the 1.5B audit file
audit_file = "probe_audit.csv"  # Adjust path if needed
# audit_file = "/content/drive/MyDrive/probe_results/probe_audit_1.5b_500.csv"

df = pd.read_csv(audit_file)

print(f"Total samples: {len(df)}")
print(f"Samples with 1000+ tokens: {(df['num_tokens'] >= 1000).sum()}")
print(f"Samples under 1000 tokens: {(df['num_tokens'] < 1000).sum()}")

# Show breakdown by correctness
print("\n--- BEFORE FILTERING ---")
print(f"All samples: {len(df)}")
print(f"  Correct: {df['is_correct'].sum()}")
print(f"  Incorrect: {(~df['is_correct']).sum()}")

long_df = df[df['num_tokens'] >= 1000]
print(f"\nLong samples (1000+ tokens): {len(long_df)}")
print(f"  Correct: {long_df['is_correct'].sum()}")
print(f"  Incorrect: {(~long_df['is_correct']).sum()}")

short_df = df[df['num_tokens'] < 1000]
print(f"\nShort samples (<1000 tokens): {len(short_df)}")
print(f"  Correct: {short_df['is_correct'].sum()}")
print(f"  Incorrect: {(~short_df['is_correct']).sum()}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df[df['is_correct']]['num_tokens'], bins=50, alpha=0.7, label='Correct', color='blue')
ax.hist(df[~df['is_correct']]['num_tokens'], bins=50, alpha=0.7, label='Incorrect', color='red')
ax.axvline(x=1000, color='black', linestyle='--', linewidth=2, label='Filter threshold (1000)')
ax.set_xlabel('Token Count')
ax.set_ylabel('Frequency')
ax.set_title('Token Distribution - BEFORE Filtering')
ax.legend()
plt.savefig('token_distribution_before_filter.png', dpi=150)
plt.show()
```

---

## Cell 4: Run Filtered Training (1.5B)

```python
# Train success probes with <1000 token filter
# Make sure you have the right paths for your data files

!python probing-llm-math/src/train_probe_filtered.py \
    --data_file probe_data_1.5b_500_fixed.pt \
    --audit_file probe_audit.csv \
    --output_dir probe_results_1.5b_filtered \
    --max_tokens 1000 \
    --test_size 0.2 \
    --seed 42
```

---

## Cell 5: Run Filtered Training (7B)

```python
# For 7B, we need to find the audit file
# If you saved it separately, use that path
# Otherwise, you may need to regenerate with token counts

# Option A: If you have a 7B audit file
!python probing-llm-math/src/train_probe_filtered.py \
    --data_file probe_data_7b_200.pt \
    --audit_file probe_audit_7b.csv \
    --output_dir probe_results_7b_filtered \
    --max_tokens 1000 \
    --test_size 0.2 \
    --seed 42

# Option B: If no audit file (will use all samples)
# !python probing-llm-math/src/train_probe_filtered.py \
#     --data_file probe_data_7b_200.pt \
#     --output_dir probe_results_7b_filtered \
#     --max_tokens 1000 \
#     --test_size 0.2 \
#     --seed 42
```

---

## Cell 6: Display Filtered Results

```python
from IPython.display import Image, display
import json
import os

# Show 1.5B filtered results
print("="*60)
print("1.5B FILTERED RESULTS")
print("="*60)

results_dir = "probe_results_1.5b_filtered"
if os.path.exists(results_dir):
    # Show heatmap
    heatmap_path = f"{results_dir}/success_probe_heatmap_filtered.png"
    if os.path.exists(heatmap_path):
        print("\nSuccess Probe Heatmap (Filtered):")
        display(Image(filename=heatmap_path))
    
    # Show token distribution
    token_path = f"{results_dir}/token_distribution_filtered.png"
    if os.path.exists(token_path):
        print("\nToken Distribution (Filtered):")
        display(Image(filename=token_path))
    
    # Show summary
    summary_path = f"{results_dir}/summary_filtered.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
else:
    print(f"Results not found at {results_dir}")

# Show 7B filtered results  
print("\n" + "="*60)
print("7B FILTERED RESULTS")
print("="*60)

results_dir = "probe_results_7b_filtered"
if os.path.exists(results_dir):
    heatmap_path = f"{results_dir}/success_probe_heatmap_filtered.png"
    if os.path.exists(heatmap_path):
        print("\nSuccess Probe Heatmap (Filtered):")
        display(Image(filename=heatmap_path))
    
    summary_path = f"{results_dir}/summary_filtered.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
else:
    print(f"Results not found at {results_dir}")
```

---

## Cell 7: Compare Filtered vs Unfiltered

```python
import json

print("="*60)
print("COMPARISON: FILTERED vs UNFILTERED")
print("="*60)

# Load unfiltered summary (adjust path)
unfiltered_1_5b = "probe_results_1.5b_500_fixed/summary.json"
filtered_1_5b = "probe_results_1.5b_filtered/summary_filtered.json"

if os.path.exists(unfiltered_1_5b) and os.path.exists(filtered_1_5b):
    with open(unfiltered_1_5b) as f:
        unf = json.load(f)
    with open(filtered_1_5b) as f:
        filt = json.load(f)
    
    print("\n1.5B Model:")
    print(f"  Samples: {unf.get('num_samples', 'N/A')} -> {filt.get('n_samples', 'N/A')}")
    
    # Compare checkpoints
    for cp in ['0pct', '50pct', '100pct']:
        unf_key = f'success_{cp}_logreg_best'
        filt_key = f'{cp}_best_acc'
        if unf_key in unf and filt_key in filt:
            unf_val = unf[unf_key]
            filt_val = filt[filt_key]
            diff = filt_val - unf_val
            print(f"  {cp}: {unf_val:.3f} -> {filt_val:.3f} ({diff:+.3f})")

# 7B comparison
unfiltered_7b = "probe_results_7b/summary.json"
filtered_7b = "probe_results_7b_filtered/summary_filtered.json"

if os.path.exists(unfiltered_7b) and os.path.exists(filtered_7b):
    with open(unfiltered_7b) as f:
        unf = json.load(f)
    with open(filtered_7b) as f:
        filt = json.load(f)
    
    print("\n7B Model:")
    print(f"  Samples: {unf.get('num_samples', 'N/A')} -> {filt.get('n_samples', 'N/A')}")
    
    for cp in ['0pct', '50pct', '100pct']:
        unf_key = f'success_{cp}_logreg_best'
        filt_key = f'{cp}_best_acc'
        if unf_key in unf and filt_key in filt:
            unf_val = unf[unf_key]
            filt_val = filt[filt_key]
            diff = filt_val - unf_val
            print(f"  {cp}: {unf_val:.3f} -> {filt_val:.3f} ({diff:+.3f})")
```

---

## Cell 8: Save to Google Drive

```python
from google.colab import drive
import shutil
import os

# Mount drive if not already
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Save filtered results
output_dir = "/content/drive/MyDrive/probe_results_filtered"
os.makedirs(output_dir, exist_ok=True)

# Copy 1.5B filtered results
src = "probe_results_1.5b_filtered"
if os.path.exists(src):
    dst = f"{output_dir}/1.5b"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    print(f"Saved 1.5B filtered results to {dst}")

# Copy 7B filtered results
src = "probe_results_7b_filtered"
if os.path.exists(src):
    dst = f"{output_dir}/7b"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    print(f"Saved 7B filtered results to {dst}")

# Copy comparison images
if os.path.exists('token_distribution_before_filter.png'):
    shutil.copy('token_distribution_before_filter.png', output_dir)
    print("Saved token distribution comparison")

print(f"\n✓ All filtered results saved to {output_dir}")
```

---

## Why This Matters

The concern is that **long responses (1000+ tokens) are almost always wrong** (2.4% accuracy from our token analysis). This could mean:

1. **Confound**: The probe might just be learning "long = wrong" from token count, not actual hidden state patterns
2. **Or**: The probe is genuinely detecting failure signals that *also* correlate with length

By filtering out 1000+ token samples:
- We remove the most extreme cases
- We test if the probe still works on "normal length" responses
- If accuracy drops significantly, length might have been confounding
- If accuracy stays similar, the probe is detecting something beyond length

---

## Expected Results

Based on our token analysis:
- ~83 samples have 1000+ tokens (mostly incorrect)
- After filtering: ~417 samples remain (1.5B)
- The balance should shift slightly toward more correct samples

**Hypothesis**: 
- If probe accuracy **drops significantly** (e.g., 81% → 65%), token length was a major confound
- If probe accuracy **stays similar** (e.g., 81% → 78%), the signal is robust

