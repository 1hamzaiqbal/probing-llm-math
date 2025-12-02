# Filtered Probe Analysis - Colab Cells

Run these cells in Colab to train probes **without** the 1000+ token samples.

---

## Cell 1: Setup & Mount Drive

```python
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Pull latest code with filtered training script
!cd probing-llm-math && git pull

# Verify the new script exists
!ls -la probing-llm-math/src/train_probe_filtered.py
```

---

## Cell 2: Check Data Files (IMPORTANT - Find Your Files!)

```python
import os

drive_root = "/content/drive/MyDrive"

print("="*60)
print("SCANNING FOR YOUR DATA FILES")
print("="*60)

# Check root Drive folder
print("\n📁 Drive root (.pt and .csv files):")
for f in sorted(os.listdir(drive_root)):
    if f.endswith('.pt') or f.endswith('.csv'):
        full_path = f"{drive_root}/{f}"
        size = os.path.getsize(full_path) / (1024*1024)  # MB
        print(f"  {f} ({size:.1f} MB)")

# Check probe_results subfolder
probe_results = f"{drive_root}/probe_results"
if os.path.exists(probe_results):
    print(f"\n📁 probe_results folder:")
    for f in sorted(os.listdir(probe_results)):
        if f.endswith('.pt') or f.endswith('.csv'):
            full_path = f"{probe_results}/{f}"
            size = os.path.getsize(full_path) / (1024*1024)
            print(f"  {f} ({size:.1f} MB)")

# Look for the specific files we need
print("\n" + "="*60)
print("CHECKING EXPECTED FILES")
print("="*60)

expected_files = [
    f"{drive_root}/probe_data_1.5b_500_fixed.pt",
    f"{drive_root}/probe_audit_500.csv",
    f"{drive_root}/probe_audit_7b_200.csv",
    f"{probe_results}/probe_data_1.5b_500_fixed.pt",
    f"{probe_results}/probe_data_7b_200.pt",
]

for path in expected_files:
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {exists} {path}")
```

---

## Cell 3: Re-evaluate 1.5B Audit File (Fix False Negatives)

```python
import pandas as pd
import importlib

# Reload evaluator with latest fixes
import src.evaluator
importlib.reload(src.evaluator)
from src.evaluator import is_equivalent

# Load the 1.5B audit file from Drive
audit_file_1_5b = "/content/drive/MyDrive/probe_audit_500.csv"
df = pd.read_csv(audit_file_1_5b)

print(f"Loaded {len(df)} samples from {audit_file_1_5b}")
print(f"Original: {df['is_correct'].sum()} correct, {(~df['is_correct']).sum()} incorrect")

# Re-evaluate all samples
changes = []
for idx, row in df.iterrows():
    gt = str(row['ground_truth'])
    pred = str(row['clean_prediction'])
    new_result = is_equivalent(pred, gt)
    
    if new_result != row['is_correct']:
        changes.append((idx, gt, pred, new_result))
        df.at[idx, 'is_correct'] = new_result

print(f"\nRe-evaluated: {len(changes)} changes")
for idx, gt, pred, result in changes[:10]:
    print(f"  Row {idx}: '{gt}' vs '{pred}' -> {'✓' if result else '✗'}")

print(f"\nAfter fix: {df['is_correct'].sum()} correct, {(~df['is_correct']).sum()} incorrect")

# Save fixed version
fixed_audit_1_5b = "/content/drive/MyDrive/probe_audit_500_fixed.csv"
df.to_csv(fixed_audit_1_5b, index=False)
print(f"\nSaved fixed audit to: {fixed_audit_1_5b}")
```

---

## Cell 4: Analyze Token Distribution BEFORE Filtering

```python
import pandas as pd
import matplotlib.pyplot as plt

# Use the FIXED 1.5B audit file
audit_file = "/content/drive/MyDrive/probe_audit_500_fixed.csv"
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

## Cell 5: Run Filtered Training (1.5B)

```python
# ⚠️ UPDATE THESE PATHS based on Cell 2 output!
# The data file should be ~large (hundreds of MB)
# The audit file should match (same number of rows as samples)

DATA_FILE_1_5B = "/content/drive/MyDrive/probe_data_1.5b_500_fixed.pt"  # UPDATE IF NEEDED
AUDIT_FILE_1_5B = "/content/drive/MyDrive/probe_audit_500_fixed.csv"    # From Cell 3

# Verify files exist before running
import os
print(f"Data file exists: {os.path.exists(DATA_FILE_1_5B)}")
print(f"Audit file exists: {os.path.exists(AUDIT_FILE_1_5B)}")

# Run filtered training
!python probing-llm-math/src/train_probe_filtered.py \
    --data_file "{DATA_FILE_1_5B}" \
    --audit_file "{AUDIT_FILE_1_5B}" \
    --output_dir probe_results_1.5b_filtered \
    --max_tokens 1000 \
    --test_size 0.2 \
    --seed 42
```

---

## Cell 6: Run Filtered Training (7B)

```python
# ⚠️ UPDATE THESE PATHS based on Cell 2 output!

DATA_FILE_7B = "/content/drive/MyDrive/probe_data_7b_200.pt"      # UPDATE IF NEEDED  
AUDIT_FILE_7B = "/content/drive/MyDrive/probe_audit_7b_200.csv"   # From Drive root

# Verify files exist before running
import os
print(f"Data file exists: {os.path.exists(DATA_FILE_7B)}")
print(f"Audit file exists: {os.path.exists(AUDIT_FILE_7B)}")

# Run filtered training
!python probing-llm-math/src/train_probe_filtered.py \
    --data_file "{DATA_FILE_7B}" \
    --audit_file "{AUDIT_FILE_7B}" \
    --output_dir probe_results_7b_filtered \
    --max_tokens 1000 \
    --test_size 0.2 \
    --seed 42
```

---

## Cell 7: Display Filtered Results

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

## Cell 8: Compare Filtered vs Unfiltered

```python
import json
import os

print("="*60)
print("COMPARISON: FILTERED vs UNFILTERED")
print("="*60)

# Load unfiltered summary from Drive
unfiltered_1_5b = "/content/drive/MyDrive/probe_results/probe_results_1.5b_500_fixed/summary.json"
filtered_1_5b = "probe_results_1.5b_filtered/summary_filtered.json"

# Try alternate path if first doesn't exist
if not os.path.exists(unfiltered_1_5b):
    unfiltered_1_5b = "/content/drive/MyDrive/probe_results_1.5b_500_fixed/summary.json"

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
else:
    print(f"1.5B: Could not find both summary files")
    print(f"  Unfiltered exists: {os.path.exists(unfiltered_1_5b)}")
    print(f"  Filtered exists: {os.path.exists(filtered_1_5b)}")

# 7B comparison
unfiltered_7b = "/content/drive/MyDrive/probe_results/probe_results_7b/summary.json"
filtered_7b = "probe_results_7b_filtered/summary_filtered.json"

if not os.path.exists(unfiltered_7b):
    unfiltered_7b = "/content/drive/MyDrive/probe_results_7b/summary.json"

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
else:
    print(f"\n7B: Could not find both summary files")
```

---

## Cell 9: Save to Google Drive

```python
import shutil
import os

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

