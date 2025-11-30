# Implementation Plan: Enhanced Probing Methodology

**Based on**: `senior_eng_notes.md` and probing literature  
**Created**: Nov 29, 2025

---

## Priority Tiers

### 🔴 HIGH PRIORITY (Must Implement)

These are low-effort, high-impact improvements that make results more rigorous and publishable.

| # | Enhancement | Why | Effort | Status |
|---|-------------|-----|--------|--------|
| 1 | **Difference-of-means probes** | Simple, interpretable baseline from Geometry of Truth | Low | ✅ DONE |
| 2 | **2D heatmap (layer × checkpoint)** | Compelling visualization of success probe evolution | Low | ✅ DONE |
| 3 | **Control task (shuffled labels)** | Validates probe isn't just overfitting high-dim space | Low | ✅ DONE |
| 4 | **Mean-pooled representations** | Alternative to last-token; cheap to capture | Medium | ⬜ TODO |
| 5 | **Proper train/val/test splits** | Tune regularization on val, report on held-out test | Medium | ⬜ TODO |

### 🟡 MEDIUM PRIORITY (Should Implement If Time)

| # | Enhancement | Why | Effort | Status |
|---|-------------|-----|--------|--------|
| 6 | **Bootstrap confidence intervals** | Error bars make results publishable | Medium | ⬜ TODO |
| 7 | **Transfer tests** | Train on easy→test on hard, or cross-topic | Medium | ⬜ TODO |
| 8 | **Calibration plots (ECE)** | Compare probe confidence to actual accuracy | Medium | ⬜ TODO |
| 9 | **Probe vs logprob baseline** | Show probes add value beyond output confidence | Medium | ⬜ TODO |

### 🟢 LOW PRIORITY (Stretch Goals)

| # | Enhancement | Why | Effort | Status |
|---|-------------|-----|--------|--------|
| 10 | **Causal interventions** | Nudge activations along probe direction | High | ⬜ TODO |
| 11 | **Multi-sample self-consistency** | Probe for "will voting help?" | High | ⬜ TODO |
| 12 | **25%/75% checkpoints** | Smoother success evolution curve | Low | ⬜ TODO |

---

## Detailed Implementation Notes

### 1. Difference-of-Means Probes

**From**: Geometry of Truth (Marks & Tegmark 2023)

Instead of logistic regression, use:
```python
w = μ_correct - μ_incorrect  # direction in activation space
b = -(μ_correct + μ_incorrect) @ w / 2  # decision boundary

# Classify by sign of (w · h + b)
pred = (X @ w + b) > 0
```

**Benefits**:
- Simpler, more interpretable
- Directly gives a "correctness direction" in representation space
- If it performs similarly to logistic regression, supports "linear representation" narrative

**Where**: Add to `train_probe.py` as alternative to logistic regression for success probe.

---

### 2. 2D Heatmap (Layer × Checkpoint)

**From**: Section 2.2 of notes, similar to Sun et al. "Probing for Arithmetic Errors"

Create a grid:
- X-axis: Checkpoint (0%, 50%, 100%)
- Y-axis: Layer (0 to 28)
- Color: Probe accuracy or AUC

**Shows**:
- Does predictive signal grow from 0% → 50% → 100%?
- Which layer band is most predictive at each stage?
- Are there checkpoints where signal is lost?

**Where**: Add to `train_probe.py` output, new plot `success_probe_heatmap_2d.png`

---

### 3. Control Task (Shuffled Labels)

**From**: Hewitt "Designing and Interpreting Probes"

**Protocol**:
1. Shuffle `is_correct` labels randomly
2. Train probe on shuffled labels
3. Report accuracy (should be ~chance = 50%)

**Why**: Confirms the probe is learning task-relevant structure, not just memorizing high-dimensional patterns.

**Where**: Add `--control` flag to `train_probe.py`

---

### 4. Mean-Pooled Representations

**From**: Section 1.1 of notes

Currently we only capture last-token activations. Add:
- **Mean over question tokens**: `mean(h[:, :prompt_len, :])`
- **Mean over answer tokens**: `mean(h[:, prompt_len:, :])`

**Comparison**: 
- Last-token vs mean-pooled
- Some tasks may prefer one over the other

**Where**: Extend `collect_probe_data.py` to save both; extend `train_probe.py` to compare.

---

### 5. Proper Train/Val/Test Splits

**From**: Section 1.2 of notes, standard ML practice

Current: Simple 80/20 train/test split  
Better: 60/20/20 train/val/test with:
1. Fit probe on train
2. Tune regularization (C for logistic, λ for ridge) on val
3. Report final metrics on held-out test

**Where**: Update `train_probe_per_layer()` in `train_probe.py`

---

### 6. Bootstrap Confidence Intervals

**From**: Section 1.2 of notes

**Protocol**:
1. Resample problems with replacement (200 bootstrap iterations)
2. Compute metric for each resample
3. Report mean ± std, or 95% CI

**Where**: Add to `train_probe.py` for final best-layer metrics

---

### 7. Transfer Tests

**From**: Section 5.1 of notes (Geometry of Truth)

**Experiments**:
1. **Cross-difficulty**: Train on levels 1-3, test on levels 4-5
2. **Cross-topic**: Train on algebra, test on number theory
3. **Cross-dataset**: Train on GSM8K, test on MATH (or vice versa)

**Why**: Tests if probe generalizes or is dataset-specific

**Where**: Add `--transfer_test` mode to `train_probe.py`

---

### 8. Calibration Plots

**From**: Section 1.2 and 3.2 of notes

**Metrics**:
- **ECE (Expected Calibration Error)**: Binned calibration error
- **Brier Score**: Mean squared error of probability estimates
- **Reliability Diagram**: Predicted prob vs actual accuracy

**Where**: Add to success probe output in `train_probe.py`

---

### 9. Probe vs Logprob Baseline

**From**: Section 3.2 of notes

**Protocol**:
1. Capture final-token logprob or average logprob during generation
2. Train baseline: `LogisticRegression(logprob) → correct`
3. Train probe: `LogisticRegression(hidden_state) → correct`
4. Train fusion: `LogisticRegression([logprob, probe_score]) → correct`
5. Compare AUC/accuracy

**Why**: Shows whether probes add value beyond surface-level confidence

**Where**: Extend `collect_probe_data.py` to save logprobs; add comparison in `train_probe.py`

---

## Implementation Order

**Phase 1** ✅ COMPLETE:
1. ✅ Difference-of-means probes - Added to `train_probe.py`
2. ✅ 2D heatmap (layer × checkpoint) - Generates `success_probe_heatmap_2d.png`
3. ✅ Control task - Generates `control_task_shuffled.png`

**Phase 2** (Next Session):
4. Mean-pooled representations
5. Proper train/val/test

**Phase 3** (If Time):
6. Bootstrap CIs
7. Transfer tests
8. Calibration plots

---

## Code Changes Summary

| File | Changes |
|------|---------|
| `train_probe.py` | Add diff-of-means, 2D heatmap, control task, proper splits |
| `collect_probe_data.py` | Add mean-pooled activations, logprobs |
| `train_probe.py` | Add bootstrap, transfer tests, calibration |

---

*Plan created: Nov 29, 2025*

