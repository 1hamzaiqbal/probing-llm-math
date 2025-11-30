# Probing LLM Math - Progress Report

**Last Updated**: December 1, 2025  
**Branch**: `colab-pipeline-setup`

---

## Project Status: Safe Checkpoint ✅

This document captures the current state of the project with all completed work. Results placeholders can be filled in from the latest runs.

---

## Completed Components

### 1. Data Collection Pipeline ✅
- `src/collect_probe_data.py`
- Collects balanced correct/incorrect samples
- Captures activations at 3 checkpoints: 0%, 50%, 100%
- **Two pooling methods**: Last token AND mean pooling
- Stores: topic, difficulty level, correctness labels
- Generates audit CSV for manual inspection
- Reproducible with `--seed` flag

### 2. Answer Evaluator ✅
- `src/evaluator.py` - 97 test cases
- Multi-strategy comparison:
  - String normalization (LaTeX cleanup)
  - Numeric equality
  - SymPy symbolic equivalence
  - Tuple/set/matrix element-wise
- Fixed 13 false negatives from 500-sample audit:
  - Shorthand fractions: `\frac12` → `\frac{1}{2}`
  - Variable prefixes: `x = 3` → `3`
  - Algebraic equivalence: `-3(x+2)(x-1)` = `-3x²-3x+6`
  - Outer parentheses: `(4x-7)` = `4x-7`

### 3. Probe Training ✅
- `src/train_probe.py`
- **Linear Probes**: Logistic Regression for classification, Ridge for regression
- **Difference-of-Means Probes**: Simple, interpretable baseline
- **MLP Probes**: 2-layer non-linear comparison (new!)
- Visualizations: 2D heatmap, PCA, pooling comparison
- Control task (shuffled labels) for validation

### 4. Stratified Evaluation ✅
- `src/stratified_eval.py`
- Generates accuracy heatmap by topic × difficulty
- Identifies "performance cliffs"

### 5. Documentation ✅
- `METHODOLOGY.md` - Full methodology writeup
- `RESULTS_TEMPLATE.md` - Template for final results
- `PROGRESS.md` - This file

---

## Latest Run: 500 Samples (1.5B Model)

### Configuration
| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-Math-1.5B-Instruct` |
| Samples | 500 (250 correct, 250 incorrect) |
| Pooling | Both (last_token + mean) |
| Checkpoints | 0%, 50%, 100% |
| Layers | 29 |
| Hidden Dim | 1536 |

### Key Results

#### Success Probes (Main Finding!)

| Checkpoint | Linear Best | Layer | DiffMeans Best | Layer |
|------------|-------------|-------|----------------|-------|
| 0% | ~69-80% | L1/L15 | ~75-79% | L15/L18 |
| 50% | ~70-71% | L8/L14 | ~60-70% | L1/L20 |
| 100% | ~77-80% | L4/L18/L23 | ~67-70% | L2/L28 |

**Key Finding**: Model predicts failure at 0% (before answering) with **75-80%** accuracy!

#### Pooling Comparison

| Checkpoint | Last Token | Mean Pool | Δ |
|------------|------------|-----------|---|
| 0% | ~69% | ~72% | +3% |
| 50% | ~71% | ~76% | +5% |
| 100% | ~77% | ~79% | +2% |

**Finding**: Mean pooling consistently outperforms last token.

#### Other Probes

| Probe | Best Score | Interpretation |
|-------|------------|----------------|
| Topic | ~82-86% | Expected (distinct vocabulary) |
| Difficulty (class) | ~45-46% | Weak (chance=20%) |
| Difficulty (regress) | R²~0.05-0.21 | Very weak |
| Control (shuffled) | ~41-52% | ✅ Passed |

---

## Model Failure Analysis (from 500-sample audit)

### Failure Distribution by Topic
- Algebra: 135 failures (54%)
- Precalculus: 49 failures (20%)
- Number Theory: 34 failures (14%)
- Counting & Probability: 32 failures (13%)

### Failure Distribution by Level
- Level 5: 132 failures (53%)
- Level 4: 48 failures (19%)
- Level 3: 40 failures (16%)
- Level 2: 26 failures (10%)
- Level 1: 4 failures (2%)

### Common Failure Modes
1. **Algebraic Errors**: Correct approach but calculation mistakes
2. **"Lazy" Guessing**: Outputs 0, 1, or -1 for complex problems (~5%)
3. **Extraction Failures**: Verbose output with buried answer
4. **Symbolic vs Numeric**: Wrong format (decimal when exact needed)

---

## File Structure

```
probing-llm-math/
├── src/
│   ├── collect_probe_data.py  # Data collection with pooling options
│   ├── train_probe.py         # Linear + MLP probe training
│   ├── evaluator.py           # 97-test robust evaluator
│   ├── model_utils.py         # Model loading (1.5B default)
│   └── stratified_eval.py     # Accuracy heatmap generation
├── test_evaluator.py          # Evaluator test suite
├── METHODOLOGY.md             # Full methodology writeup
├── RESULTS_TEMPLATE.md        # Template for results
├── PROGRESS.md                # This file
└── probe_results/             # Output directory
    ├── success_probe_*.png
    ├── pooling_comparison.png
    ├── mlp_vs_linear.png      # If --mlp used
    └── summary.json
```

---

## Commands Reference

```bash
# Collect data (Colab) - ~7 hours for 500 balanced samples
!python probing-llm-math/src/collect_probe_data.py \
    --num_samples 500 \
    --dataset math \
    --balance \
    --pooling both \
    --seed 42 \
    --output_file probe_data.pt

# Train probes with MLP comparison
!python probing-llm-math/src/train_probe.py \
    --data_file probe_data.pt \
    --output_dir probe_results \
    --mlp

# Run stratified evaluation
!python probing-llm-math/src/stratified_eval.py \
    --samples_per_cell 20 \
    --output_dir eval_results
```

---

## Bonus: Future Work

### Currently Implementing
1. **MLP Probes** ✅ - Compare linear vs non-linear
2. **Confidence Calibration** - Does probe confidence correlate with accuracy?
3. **0% → 100% Progression** - Causal analysis of how prediction changes

### Future Extensions
1. Compare 1.5B vs 7B model
2. Cross-dataset generalization (MATH → GSM8K)
3. Layer-wise analysis (why is L15-18 special?)
4. Intervention experiments (modify activations, check effect)

---

## References

- Hendrycks, D. et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset.
- Marks, S. et al. (2023). The Geometry of Truth.
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances.
