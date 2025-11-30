# Probing LLM Math - Progress Report

**Date**: November 30, 2025  
**Branch**: `colab-pipeline-setup`

## Completed Work

### 1. Data Collection Pipeline ✅
- `src/collect_probe_data.py` - Collects 150 balanced samples (75 correct, 75 incorrect)
- Captures activations at 3 checkpoints: 0% (prompt), 50% (mid-gen), 100% (final)
- Extracts **last token** hidden states at each checkpoint
- Stores topic, difficulty level, correctness labels
- Generates audit CSV for manual inspection

### 2. Evaluator ✅
- `src/evaluator.py` - Robust answer comparison using SymPy + math-verify
- `test_evaluator.py` - 48 test cases covering edge cases
- Handles: fractions, matrices, tuples, sets, base notation, scientific notation

### 3. Probe Training ✅
- `src/train_probe.py` - Trains all probe types:
  - **Success Probe**: Binary (correct/incorrect) at 0%, 50%, 100%
  - **Topic Probe**: Multi-class (algebra, precalc, counting, number_theory)
  - **Difficulty Probe**: Classification (5 levels) and Regression
- Difference-of-Means probes as baseline
- 2D heatmap (layer × checkpoint)
- Control task (shuffled labels) to validate learning
- PCA visualizations

## First Run Results (150 samples)

### Dataset Stats (First Run - 7B Model)
- Samples: 150 (75 correct, 75 incorrect)
- Topics: Algebra (87), Precalculus (27), Number Theory (19), Counting (17)
- Levels: L1 (12), L2 (27), L3 (30), L4 (36), L5 (45)
- Model: Qwen2.5-Math-7B-Instruct (4-bit)

### Model Switch (Nov 30, 2025)
**Changed to `Qwen2.5-Math-1.5B-Instruct`** for faster iteration:
- ~4x faster inference
- Lower accuracy → more balanced errors naturally
- Hidden dim: 1536 (vs 3584 for 7B)
- Same Qwen family, same prompt format

### Probe Results

| Probe | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| **Success 0%** | LogReg Best | 76.7% (L22) | Model may predict failure before answering |
| **Success 50%** | LogReg Best | 70.0% (L17) | Slight drop mid-generation |
| **Success 100%** | LogReg Best | 76.7% (L5) | Consistent with 0% |
| **Success 2D Best** | LogReg | 80.0% (L1, 100%) | Best overall |
| Topic | Accuracy | 96.7% (L12) | Topics have distinct vocabulary |
| Difficulty (class) | Accuracy | 53.3% (L20) | Weak (chance=20%) |
| Difficulty (regress) | R² | 0.361 (L8) | Weak correlation |
| Control Task | Mean Acc | 53.3% | ✅ Passed (near chance) |

### Key Observations

1. **Success Probe Works!** 76-80% accuracy suggests the model "knows" when it will fail
   - Signal exists at Layer 1 and Layers 20-22
   - Present before generation starts (0% checkpoint)

2. **Topic Probe Trivial** - 96.7% is expected because:
   - Questions contain topic-specific vocabulary ("probability", "vector", etc.)
   - NOT because topic is leaked in the prompt (verified: it's not)

3. **Difficulty Probe Weak** - The model doesn't clearly encode difficulty level
   - Could be a sample size issue (only 12 Level 1 samples)
   - Or difficulty is genuinely not represented linearly

4. **Activation Extraction**: Currently using **last token** at each position
   - Alternative: mean pooling over all tokens
   - Alternative: attention-weighted pooling

## Issues Found

1. **Evaluator False Negatives** (now fixed):
   - Base notation: `4210_{5}` vs `4210_5` 
   - Fixed in latest commit

2. **Small Sample Sizes**:
   - Counting: 17 samples
   - Level 1: 12 samples
   - Need more data for reliable results

3. **RuntimeWarning**: `invalid value encountered in divide` in diff-of-means
   - Occurs when weight vector is zero (degenerate case)
   - Non-critical, can add epsilon

## Activation Details

```
Shape: [29 layers, 3584 hidden_dim]
Position: Last token (-1) at each checkpoint
Checkpoints:
  - 0%: Last token of prompt (before any generation)
  - 50%: Last token at mid-generation
  - 100%: Last token of complete response
```

## Next Steps (Prioritized)

### High Priority
1. **Collect More Data** - 300-500 samples for statistical power
   - Balance topics more evenly
   - Ensure ≥20 samples per (topic, level) cell

2. **Try Mean Pooling** - Alternative to last-token extraction
   - May capture more distributed information
   - Compare results to last-token approach

3. **Layer Analysis** - Why does Layer 1 work well for success?
   - Visualize what information is in early vs late layers
   - Compare to middle layers (14-18)

### Medium Priority
4. **Stratified Accuracy Heatmap** - Run `stratified_eval.py`
   - Shows accuracy by topic × difficulty
   - Identifies "cliffs" where model struggles

5. **Success Probe 2D Deep Dive**
   - The 80% at (Layer 1, 100%) is interesting
   - Is Layer 1 actually meaningful or is it noise?

6. **Fine-grained Difficulty** - Treat levels as ordinal
   - Ordinal regression instead of classification
   - Or just predict "hard vs easy" (levels 1-2 vs 4-5)

### Lower Priority
7. **Baseline Comparisons**
   - Random guess baseline
   - Majority class baseline
   - Input-length-based baseline (longer Q → harder?)

8. **Cross-validation** - Current results use single train/test split
   - Add k-fold cross-validation for robust estimates

9. **Confidence Calibration** - Does the probe's confidence correlate with accuracy?

## File Structure

```
probing-llm-math/
├── src/
│   ├── collect_probe_data.py  # Data collection
│   ├── train_probe.py         # Probe training
│   ├── evaluator.py           # Answer comparison
│   ├── model_utils.py         # Model loading
│   └── stratified_eval.py     # Accuracy heatmap
├── test_evaluator.py          # Evaluator tests
├── probe_data.pt              # Collected data (150 samples)
├── probe_results/             # Training outputs
│   ├── success_probe_*.png
│   ├── topic_probe_*.png
│   ├── difficulty_probe_*.png
│   ├── pca_*.png
│   └── summary.json
└── PROGRESS.md                # This file
```

## Commands Reference

```bash
# Collect data (Colab)
!python src/collect_probe_data.py \
    --num_samples 200 \
    --dataset math \
    --balance \
    --output_file probe_data.pt

# Train probes (Colab)
!python src/train_probe.py \
    --data_file probe_data.pt \
    --output_dir probe_results

# Run stratified eval (Colab)
!python src/stratified_eval.py \
    --samples_per_cell 10 \
    --output_dir eval_results
```

