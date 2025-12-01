# Technical Report: Probing Internal Signals in Math LLMs

**Project**: Probing Internal Signals of Topic, Difficulty, and Imminent Failure in Math LLMs  
**Date**: December 2024  
**Repository**: `probing-llm-math`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Technical Architecture](#2-technical-architecture)
3. [Models & Infrastructure](#3-models--infrastructure)
4. [Dataset Details](#4-dataset-details)
5. [Probing Methodology](#5-probing-methodology)
6. [Experimental Results](#6-experimental-results)
7. [Figures Index](#7-figures-index)
8. [Ablation Studies](#8-ablation-studies)
9. [Error Analysis](#9-error-analysis)
10. [Reproducibility](#10-reproducibility)

---

## 1. Overview

### 1.1 Research Questions

1. **RQ1**: Can linear probes decode success/failure signals from LLM hidden states?
2. **RQ2**: When does this signal emerge during generation (0%, 50%, 100%)?
3. **RQ3**: Does metacognitive ability scale with model size (1.5B vs 7B)?
4. **RQ4**: Is the signal linearly encoded or does it require non-linear decoding?
5. **RQ5**: Can auxiliary signals (topic, difficulty) be decoded from the same representations?

### 1.2 Key Findings Summary

| Finding | Evidence | Significance |
|---------|----------|--------------|
| Success prediction works | 73-90% accuracy | Models have "metacognition" |
| Signal exists at 0% | 73-80% before any generation | Early failure detection possible |
| 7B > 1.5B | +7-9% improvement | Metacognition scales with size |
| Linear ≈ MLP | ±2% difference | Signal is linearly encoded |
| Token length predicts failure | r = -0.49 | Simple heuristic available |
| Topic decodable | 82-86% | Expected (distinct vocabulary) |
| Difficulty not decodable | 46-48% | Model ≠ human difficulty perception |

---

## 2. Technical Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ MATH Dataset│───▶│ Model Infer │───▶│ Hidden State│      │
│  │  (HuggingFace)   │ + Evaluation│    │  Extraction │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                            │                  │              │
│                            ▼                  ▼              │
│                    ┌─────────────┐    ┌─────────────┐       │
│                    │ probe_audit │    │ probe_data  │       │
│                    │    .csv     │    │    .pt      │       │
│                    └─────────────┘    └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      PROBE TRAINING                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Success   │    │    Topic    │    │  Difficulty │      │
│  │   Probes    │    │   Probes    │    │   Probes    │      │
│  │ (LogReg/MLP)│    │  (LogReg)   │    │(LogReg/Ridge)│     │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            ▼                                 │
│                    ┌─────────────┐                          │
│                    │  Results &  │                          │
│                    │   Figures   │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Code Organization

```
probing-llm-math/
├── src/
│   ├── model_utils.py        # Model loading, generation, hidden state capture
│   ├── evaluator.py          # Answer extraction, equivalence checking
│   ├── collect_probe_data.py # Data collection pipeline
│   ├── train_probe.py        # Probe training and evaluation
│   ├── stratified_eval.py    # Stratified accuracy heatmaps
│   ├── analyze_tokens.py     # Token length analysis
│   └── confidence_analysis.py # Calibration and trajectory analysis
├── test_evaluator.py         # Unit tests for evaluation
├── PROGRESS.md               # Results summary
├── METHODOLOGY.md            # Methodology documentation
├── PRESENTATION_GUIDE.md     # Presentation template
└── TECHREPORT.md             # This document
```

### 2.3 Data Flow

1. **Input**: MATH dataset problem + ground truth answer
2. **Generation**: Model produces solution with `\boxed{}` answer
3. **Evaluation**: Compare extracted answer to ground truth (symbolic + string)
4. **Hidden States**: Capture activations at 0%, 50%, 100% of generation
5. **Probing**: Train classifiers on frozen activations
6. **Analysis**: Generate visualizations and metrics

---

## 3. Models & Infrastructure

### 3.1 Model Specifications

| Property | Qwen2.5-Math-1.5B | Qwen2.5-Math-7B |
|----------|-------------------|-----------------|
| **Parameters** | 1.5B | 7B |
| **Layers** | 29 | 29 |
| **Hidden Dim** | 1536 | 3584 |
| **Quantization** | 4-bit (bitsandbytes) | 4-bit (bitsandbytes) |
| **Context Length** | 4096 | 4096 |
| **Max New Tokens** | 1024 | 1024 |
| **VRAM Usage** | ~4GB | ~8GB |

### 3.2 Inference Configuration

```python
generation_config = {
    "max_new_tokens": 1024,
    "do_sample": False,          # Greedy decoding
    "temperature": None,
    "top_p": None,
    "pad_token_id": tokenizer.eos_token_id,
    "output_hidden_states": True,
    "return_dict_in_generate": True
}
```

### 3.3 Hardware Requirements

- **GPU**: NVIDIA T4 (16GB) or better
- **Platform**: Google Colab Pro recommended
- **Runtime**: 
  - 1.5B: ~52 sec/sample (500 samples ≈ 7 hours)
  - 7B: ~87 sec/sample (200 samples ≈ 5 hours)

---

## 4. Dataset Details

### 4.1 Source Dataset

**MATH Benchmark** (Hendrycks et al., 2021)
- 12,500 competition mathematics problems
- 7 topics (we use 4, excluding geometry)
- 5 difficulty levels

### 4.2 Filtered Dataset

| Topic | Original | After Filtering |
|-------|----------|-----------------|
| Algebra | 1744 | 1744 |
| Precalculus | 869 | 869 |
| Number Theory | 869 | 869 |
| Counting & Probability | 771 | 771 |
| **Geometry** | 870 | **Excluded** |
| Intermediate Algebra | 903 | Merged with Algebra |
| Prealgebra | 871 | Merged with Algebra |

**Exclusion Rationale**: Geometry problems often require visual reasoning that text-only models cannot perform reliably.

### 4.3 Sample Distribution (1.5B, 500 samples)

#### By Topic
| Topic | Count | Percentage |
|-------|-------|------------|
| Algebra | 310 | 62.0% |
| Precalculus | 72 | 14.4% |
| Number Theory | 68 | 13.6% |
| Counting & Probability | 50 | 10.0% |

#### By Difficulty Level
| Level | Count | Percentage |
|-------|-------|------------|
| Level 1 | 45 | 9.0% |
| Level 2 | 76 | 15.2% |
| Level 3 | 98 | 19.6% |
| Level 4 | 105 | 21.0% |
| Level 5 | 176 | 35.2% |

#### By Correctness
| Status | Count | Percentage |
|--------|-------|------------|
| Correct | 261 | 52.2% |
| Incorrect | 239 | 47.8% |

### 4.4 Sample Distribution (7B, 200 samples)

#### By Topic
| Topic | Count | Percentage |
|-------|-------|------------|
| Algebra | 125 | 62.5% |
| Precalculus | 33 | 16.5% |
| Number Theory | 20 | 10.0% |
| Counting & Probability | 22 | 11.0% |

#### By Correctness
| Status | Count | Percentage |
|--------|-------|------------|
| Correct | 100 | 50.0% |
| Incorrect | 100 | 50.0% |

---

## 5. Probing Methodology

### 5.1 Activation Checkpoints

| Checkpoint | Description | Tokens Included |
|------------|-------------|-----------------|
| **0%** | Question-only | System prompt + user question |
| **50%** | Mid-generation | Question + first half of response |
| **100%** | Full response | Question + complete response |

**Implementation**: 
- 0%: Forward pass on prompt only
- 50%: Replay truncated response (first 50% of tokens)
- 100%: Full generation with hidden state capture

### 5.2 Activation Extraction

```python
# Last token activation (default)
activation = hidden_states[-1][:, -1, :]  # [1, hidden_dim]

# Mean pooling (alternative)
activation = hidden_states[-1].mean(dim=1)  # [1, hidden_dim]
```

**Shape**: `[num_layers, hidden_dim]` per sample
- 1.5B: `[29, 1536]`
- 7B: `[29, 3584]`

### 5.3 Probe Architectures

#### Logistic Regression (Primary)
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
```

#### Difference-of-Means (Interpretable)
```python
# Compute class centroids
mean_correct = X[y == 1].mean(axis=0)
mean_incorrect = X[y == 0].mean(axis=0)

# Direction vector
w = mean_correct - mean_incorrect
w = w / np.linalg.norm(w)

# Classify by projection
predictions = (X @ w > threshold).astype(int)
```

#### MLP (Non-linear comparison)
```python
MLPClassifier(
    hidden_layer_sizes=(256,),
    max_iter=1000,
    random_state=42,
    early_stopping=True
)
```

#### Ridge Regression (Difficulty)
```python
Ridge(alpha=1.0)
```

### 5.4 Evaluation Protocol

- **Train/Test Split**: 80/20 stratified
- **Metric (Classification)**: Accuracy
- **Metric (Regression)**: R² score
- **Cross-validation**: Single split (limited compute)

---

## 6. Experimental Results

### 6.1 Success Probe Results

#### Table 1: Success Probe Accuracy - 1.5B Model (500 samples)

| Checkpoint | LogReg Best | Layer | DiffMeans Best | Layer |
|------------|-------------|-------|----------------|-------|
| 0% | 73.0% | L17 | 72.0% | L16 |
| 50% | 73.0% | L25 | 69.0% | L28 |
| 100% | 81.0% | L18 | 72.0% | L2 |

#### Table 2: Success Probe Accuracy - 7B Model (200 samples)

| Checkpoint | LogReg Best | Layer | DiffMeans Best | Layer |
|------------|-------------|-------|----------------|-------|
| 0% | 80.0% | L4 | 80.0% | L18 |
| 50% | 72.5% | L25 | 75.0% | L26 |
| 100% | 90.0% | L22 | 85.0% | L28 |

#### Table 3: Model Comparison Summary

| Metric | 1.5B | 7B | Δ |
|--------|------|-----|-----|
| Success @ 0% | 73% | 80% | +7% |
| Success @ 50% | 73% | 72.5% | -0.5% |
| Success @ 100% | 81% | 90% | +9% |
| Topic | 86% | 82.5% | -3.5% |
| Difficulty (class) | 46% | 47.5% | +1.5% |
| Difficulty (R²) | 0.209 | 0.018 | -0.191 |

### 6.2 MLP vs Linear Comparison (1.5B)

#### Table 4: MLP vs Logistic Regression

| Checkpoint | Linear Best | MLP Best | Δ | Winner |
|------------|-------------|----------|------|--------|
| 0% | 73.0% | 72.0% | -1.0% | Linear |
| 50% | 73.0% | 75.0% | +2.0% | MLP |
| 100% | 81.0% | 80.0% | -1.0% | Linear |

**Interpretation**: No significant benefit from non-linear probes. Success signal is linearly encoded.

### 6.3 Pooling Method Comparison (1.5B)

#### Table 5: Last Token vs Mean Pooling

| Checkpoint | Last Token | Mean Pool | Winner |
|------------|------------|-----------|--------|
| 0% | 73.0% | 70.0% | Last Token |
| 50% | 73.0% | 78.0% | Mean Pool |
| 100% | 81.0% | 78.0% | Last Token |

**Interpretation**: Mixed results. Mean pooling better at 50% (aggregates more context), last token better at endpoints.

### 6.4 Topic Probe Results

#### Table 6: Topic Classification (0% checkpoint)

| Model | Best Accuracy | Best Layer |
|-------|---------------|------------|
| 1.5B | 86.0% | L14 |
| 7B | 82.5% | L12 |
| Baseline (majority) | ~62% | - |

#### Table 7: Per-Topic Classification Report (1.5B, L14)

| Topic | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Algebra | 0.92 | 0.90 | 0.91 | 62 |
| Counting & Prob | 0.88 | 0.70 | 0.78 | 10 |
| Number Theory | 0.60 | 0.86 | 0.71 | 14 |
| Precalculus | 1.00 | 0.79 | 0.88 | 14 |
| **Weighted Avg** | 0.88 | 0.86 | 0.86 | 100 |

### 6.5 Difficulty Probe Results

#### Table 8: Difficulty Probes (0% checkpoint)

| Model | Classification | Layer | Regression R² | Layer |
|-------|----------------|-------|---------------|-------|
| 1.5B | 46.0% | L8 | 0.209 | L16 |
| 7B | 47.5% | L9 | 0.018 | L21 |
| Baseline (random) | 20% | - | 0.0 | - |

**Interpretation**: Difficulty probes perform above chance but poorly overall. Models don't encode human-assigned difficulty well.

### 6.6 Control Task Results

#### Table 9: Shuffled Label Control Task

| Model | Mean Accuracy | Expected | Status |
|-------|---------------|----------|--------|
| 1.5B | 50.3% | 50% | ✅ Passed |
| 7B | 47.3% | 50% | ✅ Passed |

**Interpretation**: Probes achieve chance-level on shuffled labels, confirming they learn real patterns, not memorization.

### 6.7 Token Length Analysis (1.5B)

#### Table 10: Token Statistics by Correctness

| Metric | Correct | Incorrect | Difference |
|--------|---------|-----------|------------|
| Mean | 458 | 731 | +273 (60%) |
| Median | 400 | 777 | +377 |
| Std Dev | 206 | 280 | +74 |
| Min | 137 | 137 | - |
| Max | 1024 | 1024 | - |

#### Table 11: Statistical Test

| Test | Value | Interpretation |
|------|-------|----------------|
| t-statistic | -12.40 | Highly significant |
| p-value | < 0.0001 | Reject null hypothesis |
| Pearson r | -0.486 | Moderate negative correlation |

#### Table 12: Accuracy by Token Range

| Token Range | Accuracy | n | Interpretation |
|-------------|----------|---|----------------|
| ≤253 (short) | 76.0% | 50 | Good |
| 317-403 (optimal) | 79.5% | 73 | Best |
| ≥1024 (long) | 2.4% | 83 | Very poor |

**Interpretation**: Long responses strongly predict failure. Model "rambles" when uncertain.

### 6.8 Confidence Calibration (1.5B)

#### Table 13: Expected Calibration Error (ECE)

| Checkpoint | ECE | Interpretation |
|------------|-----|----------------|
| 0% | 0.228 | Moderate |
| 50% | 0.290 | Poor |
| 100% | 0.174 | Best |

#### Table 14: Layer-wise Calibration

| Metric | Layer | Value |
|--------|-------|-------|
| Best Calibrated | L0 | ECE = 0.056 |
| Most Accurate | L18 | Acc = 81.0% |

#### Table 15: Prediction Trajectory Categories

| Category | Count | Percentage |
|----------|-------|------------|
| Stable Correct | 243 | 48.6% |
| Stable Incorrect | 221 | 44.2% |
| Other | 15 | 3.0% |
| Degradation | 7 | 1.4% |
| Recovery | 6 | 1.2% |
| False Degradation | 5 | 1.0% |
| False Recovery | 3 | 0.6% |

**Interpretation**: 93% of predictions are stable across checkpoints. Probe rarely changes its mind.

### 6.9 Stratified Model Accuracy (1.5B)

#### Table 16: Overall Performance

| Metric | Value |
|--------|-------|
| Total Problems | 400 |
| Correct | 290 |
| Accuracy | 72.5% |

---

## 7. Figures Index

### 7.1 Success Probe Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `success_probe_accuracy.png` | Line plot of accuracy by layer for each checkpoint | `probe_results_*/` | Shows layer 15-20 is "metacognition zone" |
| `success_probe_heatmap_2d.png` | 2D heatmap (Layer × Checkpoint) | `probe_results_*/` | Visual summary of all results; green = high accuracy |
| `control_task_shuffled.png` | Shuffled label control | `probe_results_*/` | Flat line at 50% confirms no memorization |

**Interpretation (Heatmap)**: 
- Horizontal axis: Checkpoints (0%, 50%, 100%)
- Vertical axis: Layers (0-28)
- Color: Accuracy (red=low, green=high)
- Pattern: Green intensifies at 100% checkpoint, especially in middle layers

### 7.2 MLP Comparison Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `mlp_vs_linear.png` | Side-by-side Linear vs MLP accuracy | `probe_results_mlp/` | Lines overlap → linear is sufficient |

**Interpretation**: 
- Blue = Linear (LogReg)
- Red = MLP
- Nearly identical curves across all checkpoints
- Confirms success signal is linearly encoded

### 7.3 Pooling Comparison Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `pooling_comparison.png` | Last token vs mean pooling accuracy | `probe_results_*/` | Mixed results by checkpoint |

**Interpretation**:
- Blue = Last Token
- Green = Mean Pool
- 50% checkpoint favors mean pooling (more context helps)
- 0% and 100% favor last token

### 7.4 Topic Probe Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `topic_probe_accuracy.png` | Topic classification accuracy by layer | `probe_results_*/` | Peaks at layers 12-14, ~86% accuracy |

**Interpretation**: 
- High accuracy expected (questions have distinctive vocabulary)
- Middle layers encode semantic content best

### 7.5 Difficulty Probe Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `difficulty_probe_classification.png` | 5-class difficulty accuracy by layer | `probe_results_*/` | Low accuracy (~46%), barely above baseline |
| `difficulty_probe_regression.png` | Ridge regression R² by layer | `probe_results_*/` | Low R² (~0.2), model ≠ human difficulty |

**Interpretation**:
- Models don't perceive difficulty like humans
- Possibly because "difficulty" is defined by human competition results, not intrinsic problem properties

### 7.6 PCA Visualization Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `pca_layer20_correctness.png` | PCA colored by correct/incorrect | `probe_results_*/` | Partial separation visible |
| `pca_layer20_topic.png` | PCA colored by topic | `probe_results_*/` | Topic clusters somewhat visible |
| `pca_layer20_difficulty.png` | PCA colored by difficulty | `probe_results_*/` | No clear difficulty clustering |

**Interpretation (Correctness PCA)**:
- Blue = Correct, Red = Incorrect
- Some separation but not clean → explains 73-81% accuracy, not 100%
- PC1 explains ~10-11% variance

### 7.7 Token Analysis Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `token_distribution.png` | Histogram + boxplot by correctness | `token_analysis/` | Incorrect skews right (longer) |
| `accuracy_by_tokens.png` | Accuracy vs token count bins | `token_analysis/` | Accuracy drops sharply after 800 tokens |
| `tokens_by_topic_level.png` | Heatmap of tokens by topic×level | `token_analysis/` | Harder problems → longer responses |
| `token_trend.png` | Rolling accuracy vs token count | `token_analysis/` | Smooth decline with length |

**Interpretation (Distribution)**:
- Two distinct distributions with some overlap
- Correct: centered ~400-500 tokens
- Incorrect: spread 600-1000+ tokens
- Clear heuristic: if response > 800 tokens, likely wrong

### 7.8 Confidence Analysis Figures

| Filename | Description | Location | Key Insight |
|----------|-------------|----------|-------------|
| `calibration_curves.png` | Reliability diagrams per checkpoint | `confidence_analysis/` | 100% checkpoint is best calibrated |
| `confidence_trajectories.png` | Individual sample trajectories | `confidence_analysis/` | Most are flat lines (stable) |
| `trajectory_categories.png` | Bar chart of trajectory types | `confidence_analysis/` | 93% stable |
| `confidence_change_prediction.png` | Δconfidence as predictor | `confidence_analysis/` | Δconf not useful (0.087 correlation) |
| `layer_calibration.png` | ECE by layer | `confidence_analysis/` | Early layers better calibrated |

**Interpretation (Calibration Curves)**:
- Diagonal = perfect calibration
- 0% and 50% curves deviate more
- 100% curve closer to diagonal
- Model becomes more calibrated as it generates

---

## 8. Ablation Studies

### 8.1 Layer Selection

| Experiment | Result |
|------------|--------|
| Best success layer (1.5B) | L17-18 |
| Best success layer (7B) | L4 (0%), L22 (100%) |
| Best topic layer | L12-14 |
| Best difficulty layer | L8-9 |

**Finding**: Middle-to-late layers (50-70% depth) encode success best.

### 8.2 Sample Size Effect

| Samples | 1.5B Success @ 100% |
|---------|---------------------|
| 200 | ~80% |
| 500 | 81% |

**Finding**: Marginal improvement with more samples. 200 sufficient for signal.

### 8.3 Balance Effect

Balanced sampling (50/50 correct/incorrect) vs natural distribution:
- Natural: ~72% correct (1.5B on MATH)
- Balanced: Forces model to see equal failure cases

**Finding**: Balanced sampling necessary for meaningful probe training.

---

## 9. Error Analysis

### 9.1 Evaluation Challenges

| Issue | Solution | Files Affected |
|-------|----------|----------------|
| LaTeX formatting (`\dfrac` vs `\frac`) | Normalization | `evaluator.py` |
| Symbolic equivalence | SymPy parsing | `evaluator.py` |
| Unit stripping (`575 students` vs `575`) | Regex extraction | `evaluator.py` |
| Derivation extraction (`...=D`) | Final value parser | `evaluator.py` |
| Base notation (`4210_{5}` vs `4210_5`) | Subscript handling | `evaluator.py` |

### 9.2 False Negative Examples (Fixed)

| Ground Truth | Model Output | Issue |
|--------------|--------------|-------|
| `\dfrac{1}{128}` | `\frac{1}{128}` | LaTeX variant |
| `575\text{ students}` | `575` | Unit suffix |
| `\frac 59` | `\frac{5}{9}` | Spacing variant |
| `D` | `...= D` | Derivation extraction |

### 9.3 Remaining Limitations

1. **Complex symbolic expressions**: Some algebraic equivalences not detected
2. **Set/tuple ordering**: May miss equivalent reorderings
3. **Numeric precision**: Float comparison edge cases

---

## 10. Reproducibility

### 10.1 Random Seeds

| Component | Seed |
|-----------|------|
| Data collection | 42 |
| Train/test split | 42 |
| Model generation | Deterministic (no sampling) |

### 10.2 Key Commands

```bash
# Data Collection (1.5B, 500 samples)
python src/collect_probe_data.py \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --num_samples 500 \
    --balance \
    --pooling both \
    --seed 42 \
    --output probe_data_1.5b_500.pt

# Data Collection (7B, 200 samples)
python src/collect_probe_data.py \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --num_samples 200 \
    --balance \
    --pooling last_token \
    --seed 42 \
    --output probe_data_7b_200.pt

# Probe Training (with MLP)
python src/train_probe.py \
    --data_file probe_data_1.5b_500.pt \
    --output_dir probe_results_1.5b \
    --mlp

# Token Analysis
python src/analyze_tokens.py \
    --audit_file probe_audit.csv \
    --output_dir token_analysis

# Confidence Analysis
python src/confidence_analysis.py \
    --data_file probe_data_1.5b_500.pt \
    --layer 15 \
    --output_dir confidence_analysis

# Stratified Evaluation
python src/stratified_eval.py \
    --samples_per_cell 20 \
    --output_dir eval_results
```

### 10.3 Dependencies

```
torch>=2.0
transformers>=4.35
datasets
bitsandbytes
scikit-learn
pandas
matplotlib
seaborn
sympy
scipy
tqdm
```

### 10.4 Output Files

| File | Contents |
|------|----------|
| `probe_data_*.pt` | Hidden states, labels, metadata |
| `probe_audit.csv` | Per-sample predictions for debugging |
| `summary.json` | Aggregated metrics |
| `*.png` | Visualization figures |

---

## Appendix A: Data Schema

### probe_data.pt Structure

```python
{
    'activations_0pct': Tensor[N, L, H],      # Question-only activations
    'activations_50pct': Tensor[N, L, H],     # Mid-generation activations
    'activations_100pct': Tensor[N, L, H],    # Full response activations
    'activations_0pct_mean': Tensor[N, L, H], # Mean-pooled (if enabled)
    'activations_50pct_mean': Tensor[N, L, H],
    'activations_100pct_mean': Tensor[N, L, H],
    'labels': Tensor[N],                       # 0=incorrect, 1=correct
    'topics': List[str],                       # Topic labels
    'levels': List[int],                       # Difficulty 1-5
    'metadata': {
        'num_samples': int,
        'num_layers': int,
        'hidden_dim': int,
        'pooling_method': str,
        'dataset': str,
        'seed': int,
        'has_50pct': bool,
        'has_mean_pooling': bool,
        'has_last_token': bool
    }
}
```

### probe_audit.csv Columns

| Column | Type | Description |
|--------|------|-------------|
| `idx` | int | Sample index |
| `question` | str | Problem text |
| `ground_truth` | str | Correct answer |
| `prediction` | str | Model's answer |
| `is_correct` | bool | Evaluation result |
| `topic` | str | Math topic |
| `level` | int | Difficulty 1-5 |
| `num_tokens` | int | Response length |

---

## Appendix B: Statistical Tests

### Token Length t-test

```
H0: μ_correct = μ_incorrect
H1: μ_correct ≠ μ_incorrect

t = -12.40
df = 498
p < 0.0001

Conclusion: Reject H0. Significant difference in token length.
```

### Effect Size (Cohen's d)

```
d = (458 - 731) / pooled_std
d ≈ -1.1 (large effect)
```

---

*End of Technical Report*

