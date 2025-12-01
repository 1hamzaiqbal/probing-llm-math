# Probing LLM Math - Final Results

**Last Updated**: December 1, 2025  
**Branch**: `colab-pipeline-setup`  
**Status**: ✅ Complete

---

## Executive Summary

We trained linear probes on `Qwen2.5-Math-1.5B-Instruct` to decode internal signals about math problem success, topic, and difficulty. Key findings:

1. **Success Prediction**: 73-81% accuracy - model "knows" it will fail before answering
2. **Linear ≈ MLP**: Success signal is **linearly decodable** (no benefit from non-linear probes)
3. **Token Length Matters**: Incorrect answers are 60% longer (731 vs 458 tokens)
4. **Topic Classification**: 86% accuracy (expected - distinct vocabulary)
5. **Difficulty Estimation**: 46% (weak signal)
6. **Model Accuracy**: 72.5% overall on stratified MATH dataset

---

## Dataset & Model

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-Math-1.5B-Instruct` (4-bit) |
| Probe Samples | 500 (261 correct, 239 incorrect) |
| Eval Samples | 400 (20 per topic×level cell) |
| Layers | 29 |
| Hidden Dim | 1536 |

### Topic Distribution (Probe Data)
| Topic | Count |
|-------|-------|
| Algebra | 310 |
| Precalculus | 72 |
| Number Theory | 68 |
| Counting & Probability | 50 |

### Level Distribution
| Level | Count |
|-------|-------|
| Level 1 | 45 |
| Level 2 | 76 |
| Level 3 | 98 |
| Level 4 | 105 |
| Level 5 | 176 |

---

## 1. Success Probes (Main Finding!)

### Results by Checkpoint

| Checkpoint | LogReg Best | Layer | DiffMeans Best | Layer |
|------------|-------------|-------|----------------|-------|
| **0%** | 73.0% | L17 | 72.0% | L16 |
| **50%** | 73.0% | L25 | 69.0% | L28 |
| **100%** | **81.0%** | L18 | 72.0% | L2 |

### 2D Heatmap (Layer × Checkpoint)
- **Best LogReg**: Layer 22, 100% = **80.0%**
- **Best DiffMeans**: Layer 16, 0% = **78.0%**

### Interpretation
The model has **"metacognition"** - it encodes success/failure signal before generating any answer (73% at 0%). This signal strengthens as generation progresses, reaching 81% at 100%.

### Control Task
- Mean accuracy: **50.3%** (expected ~50%)
- ✅ **Passed** - confirms probes learn real patterns, not memorizing

---

## 2. Token Length Analysis (New Finding!)

### Key Stats
| Metric | Correct | Incorrect |
|--------|---------|-----------|
| Mean tokens | **458** | **731** |
| Median tokens | 400 | 777 |

### Statistical Test
- **t-statistic**: -12.40
- **p-value**: < 0.0001
- **Correlation** (tokens vs correct): **-0.486**

### Accuracy by Token Range
| Token Range | Accuracy | n |
|-------------|----------|---|
| 317-403 (optimal) | **79.5%** | 73 |
| ≤253 (short) | 76.0% | 50 |
| ≥1024 (long) | **2.4%** | 83 |

**Insight**: Long responses strongly predict failure. Model "rambles" when uncertain.

---

## 3. Pooling Method Comparison

| Checkpoint | Last Token | Mean Pool | Winner |
|------------|------------|-----------|--------|
| 0% | **73.0%** | 70.0% | Last Token |
| 50% | 73.0% | **78.0%** | Mean Pool |
| 100% | **81.0%** | 78.0% | Last Token |

**Mixed results**: Mean pooling wins at 50% (+5%), but last token wins at 0% and 100%.

---

## 4. MLP vs Linear Probes (Key Finding!)

### Results by Checkpoint

| Checkpoint | Linear Best | MLP Best | Δ | Winner |
|------------|-------------|----------|------|--------|
| **0%** | 73.0% | 72.0% | -1.0% | Linear |
| **50%** | 73.0% | 75.0% | +2.0% | MLP |
| **100%** | **81.0%** | 80.0% | -1.0% | Linear |

### Interpretation

**The success signal is linearly encoded!** 

MLP probes (with hidden layer) perform nearly identically to simple logistic regression (within ±2%). This means:
- The model represents success/failure in a **linear subspace**
- No complex non-linear transformations needed to decode it
- Consistent with the "linear representation hypothesis" in interpretability research

This is a **cleaner story** for the paper: simple linear probes are sufficient to extract the metacognitive signal.

---

## 5. Confidence Calibration

### Expected Calibration Error (ECE)
| Checkpoint | ECE | Interpretation |
|------------|-----|----------------|
| 0% | 0.228 | Moderate calibration |
| 50% | 0.290 | Worse calibration |
| 100% | **0.174** | Best calibration |

### Layer-wise Calibration
- **Best calibrated**: Layer 0 (ECE = 0.056)
- **Most accurate**: Layer 18 (Acc = 81.0%)

### Prediction Trajectories
| Category | % of Samples |
|----------|--------------|
| Stable Correct | 48.6% |
| Stable Incorrect | 44.2% |
| Recovery (wrong→right) | 1.2% |
| Degradation (right→wrong) | 1.4% |
| False Recovery | 0.6% |
| False Degradation | 1.0% |

**93% of predictions are stable** - probe rarely changes its mind during generation.

### Confidence Change as Predictor
- Correlation (Δconf vs correct): **0.087** (weak)
- Predict from Δconf: 55.4%
- Predict from final conf: **95.4%**

**Conclusion**: Final confidence is highly predictive; change in confidence is not.

---

## 6. Topic Probes

| Metric | Value |
|--------|-------|
| Best Accuracy | **86.0%** |
| Best Layer | L14 |
| Baseline (majority) | ~62% |

### Classification Report (L14)
| Topic | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Algebra | 0.92 | 0.90 | 0.91 |
| Counting & Prob | 0.88 | 0.70 | 0.78 |
| Number Theory | 0.60 | 0.86 | 0.71 |
| Precalculus | 1.00 | 0.79 | 0.88 |

---

## 7. Difficulty Probes

| Mode | Best Score | Best Layer |
|------|------------|------------|
| Classification (5-class) | **46.0%** | L8 |
| Regression | R² = **0.209** | L16 |
| Baseline (random) | 20% | - |

**Interpretation**: Model doesn't strongly encode human-assigned difficulty levels.

---

## 8. Stratified Model Accuracy

### Overall
- **Accuracy**: 290/400 = **72.5%**
- **Top Cliff**: Precalculus L4→L5 (25% drop)

### By Topic (estimated from 72.5% overall)
Performance degrades significantly at Level 5 across all topics, with precalculus showing the steepest cliff.

---

## Key Insights

### 1. Success Prediction Works
- 73% at 0% checkpoint = model "knows" before answering
- 81% at 100% = strong post-hoc signal
- Layer 16-18 is the "metacognition" zone

### 2. Linear Probes Are Sufficient
- MLP probes perform within ±2% of linear probes
- Success signal is **linearly encoded** in hidden states
- Supports the "linear representation hypothesis"
- Simpler model = cleaner interpretation

### 3. Token Length is Highly Predictive
- **Correlation -0.49** with correctness
- Long responses (>1024 tokens) have only 2.4% accuracy
- Optimal range: 317-403 tokens (79.5% accuracy)
- **Practical implication**: Can use response length as a simple failure detector

### 4. Calibration Improves with Generation
- ECE drops from 0.228 (0%) to 0.174 (100%)
- Model becomes more calibrated as it sees its own output

### 5. Topic is Easy, Difficulty is Hard
- Topic: 86% (questions have distinctive vocabulary)
- Difficulty: 46% (model doesn't perceive difficulty like humans)

### 6. Stable Trajectories
- 93% of predictions don't change during generation
- Recovery/degradation is rare

---

## File Outputs

### Probe Results
```
probe_results_1.5b_500_fixed/
├── success_probe_accuracy.png
├── success_probe_heatmap_2d.png
├── mlp_vs_linear.png          # NEW: MLP comparison
├── control_task_shuffled.png
├── topic_probe_accuracy.png
├── difficulty_probe_classification.png
├── difficulty_probe_regression.png
├── pooling_comparison.png
├── pca_layer20_*.png
└── summary.json
```

### Token Analysis
```
token_analysis/
├── token_distribution.png
├── accuracy_by_tokens.png
├── tokens_by_topic_level.png
└── token_trend.png
```

### Confidence Analysis
```
confidence_analysis/
├── calibration_curves.png
├── confidence_trajectories.png
├── trajectory_categories.png
├── confidence_change_prediction.png
├── layer_calibration.png
└── summary.json
```

### Stratified Eval
```
eval_results_1.5b/
├── accuracy_heatmap.png
├── results.pt
└── summary.txt
```

---

## Commands Reference

```bash
# Data collection
python src/collect_probe_data.py --num_samples 500 --balance --pooling both

# Probe training (linear)
python src/train_probe.py --data_file probe_data.pt --output_dir probe_results

# Probe training (with MLP comparison)
python src/train_probe.py --data_file probe_data.pt --output_dir probe_results --mlp

# Token analysis
python src/analyze_tokens.py --audit_file probe_audit.csv

# Confidence analysis
python src/confidence_analysis.py --data_file probe_data.pt --layer 15

# Stratified evaluation
python src/stratified_eval.py --samples_per_cell 20 --output_dir eval_results
```
