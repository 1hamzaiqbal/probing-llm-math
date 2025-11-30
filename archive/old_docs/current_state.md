# Current State & Gameplan — probing-llm-math

**Last Updated**: Nov 29, 2025  
**Branch**: `colab-pipeline-setup`  
**Active Notebook**: `pipeline_colab.ipynb`

---

## 1. Project Goal Recap (from proposal)

Train linear probes on frozen activations of a math-specialized 7B LLM to test:

| Probe | Description | Input Stage |
|-------|-------------|-------------|
| **O2: Topic** | Classify topic (algebra/number theory/combinatorics) | Question-only (0%) |
| **O3: Success** | Predict eventual correctness | 0% (question-only), 50% (mid-solution) |
| **O4: Difficulty** | Predict human difficulty (MATH levels 1–5) | Question-only (0%) |

Additional goal: Generate **accuracy heatmaps** (topic × difficulty) to identify "cliffs".

---

## 2. Current Architecture

### 2.1 Chosen Model
```
Qwen/Qwen2.5-Math-7B-Instruct (4-bit quantized via bitsandbytes)
```
- **Why**: Strong MATH/GSM8K scores, HF-friendly, direct answer prompting works well
- **VRAM**: ~6-8GB with 4-bit on T4; fits comfortably in Colab

### 2.2 Core Modules (`src/`) — ALL UPDATED ✅

| File | Status | Purpose |
|------|--------|---------|
| `model_utils.py` | ✅ Working | Load model (4-bit), generate with hidden states |
| `evaluator.py` | ✅ **Upgraded** | Robust answer verification (45 test cases) |
| `collect_probe_data.py` | ✅ **Upgraded** | Full metadata + 0%/50%/100% activations |
| `train_probe.py` | ✅ **Upgraded** | All 3 probe types with visualizations |
| `stratified_eval.py` | ✅ **New** | Heatmap generation, cliff detection |

### 2.3 Data Sources

| Source | Script/Location | Status |
|--------|-----------------|--------|
| MATH dataset (HF) | `EleutherAI/hendrycks_math` | ✅ Used in all scripts |
| GSM8K | HF `gsm8k/main` | ✅ Alternate option |
| Custom stratified | `data/problems.csv` (75 problems) | ✅ Available |

---

## 3. What's Complete ✅

### 3.1 Evaluator (`evaluator.py`)

**45 test cases passing.** Handles:
- Fraction notation (`2/5` ↔ `\frac{2}{5}`)
- Rationalized denominators (`1/\sqrt{2}` ↔ `\sqrt{2}/2`)
- Complex numbers (`i/5` ↔ `(1/5)i`)
- Matrices (element-wise comparison)
- Coordinate tuples (`(1, \frac{9}{2})` ↔ `(1, 4.5)`)
- Numbers with thousands separators (`90,900,909`)
- Sets (unordered comparison)
- Intervals with infinity notation

Run `python test_evaluator.py` to verify.

### 3.2 Data Collection (`collect_probe_data.py`)

**New data schema:**
```python
{
    "problem_id": str,
    "topic": str,           # algebra | number_theory | counting_probability | precalculus
    "level_num": int,       # 1-5
    "level_str": str,       # "Level 3"
    "question": str,
    "ground_truth": str,
    "prediction": str,
    "is_correct": bool,
    "num_response_tokens": int,
    
    # Activations [num_layers+1, hidden_dim]
    "activations_0pct": Tensor,    # Question-only (before generation)
    "activations_50pct": Tensor,   # Mid-generation (optional)
    "activations_100pct": Tensor,  # Full response
    
    # Legacy compatibility
    "prompt_activations": Tensor,
    "response_activations": Tensor,
}
```

**Usage:**
```bash
python src/collect_probe_data.py \
    --num_samples 200 \
    --dataset math \
    --balance \
    --output_file probe_data.pt \
    --audit_file probe_audit.csv
```

**New flags:**
- `--no_50pct`: Skip 50% checkpoint capture (faster)
- `--include_geometry`: Include geometry problems (excluded by default)
- `--level 3`: Filter to specific difficulty level

### 3.3 Probe Training (`train_probe.py`)

**Trains all three probe types:**

1. **Success Probe** (binary)
   - Checkpoints: 0%, 50%, 100%
   - Outputs: `success_probe_accuracy.png`

2. **Topic Probe** (multi-class)
   - Uses 0% checkpoint
   - Outputs: `topic_probe_accuracy.png`

3. **Difficulty Probe** (classification + regression)
   - Uses 0% checkpoint
   - Outputs: `difficulty_probe_classification.png`, `difficulty_probe_regression.png`

**Also generates:**
- PCA visualizations (by correctness, topic, difficulty)
- `summary.json` with best layers and scores

**Usage:**
```bash
python src/train_probe.py \
    --data_file probe_data.pt \
    --output_dir probe_results
```

### 3.4 Stratified Evaluation (`stratified_eval.py`)

**Generates accuracy heatmaps (topic × difficulty):**

**Outputs:**
- `accuracy_heatmap.png`: Visual heatmap
- `summary.txt`: Text summary with cliff detection
- `eval_log.csv`: Detailed per-problem results
- `results.pt`: Full results for re-analysis

**Usage:**
```bash
python src/stratified_eval.py \
    --samples_per_cell 5 \
    --output_dir eval_results
```

Can also reload existing results:
```bash
python src/stratified_eval.py \
    --results_file eval_results/results.pt \
    --output_dir eval_results
```

---

## 4. Quick Start (Colab)

### Step 1: Setup & Load Model
```python
# Run cells 1-4 in pipeline_colab.ipynb
```

### Step 2: Collect Probe Data
```bash
!python probing-llm-math/src/collect_probe_data.py \
    --num_samples 100 \
    --dataset math \
    --balance \
    --output_file probe_data.pt
```

### Step 3: Train Probes
```bash
!python probing-llm-math/src/train_probe.py \
    --data_file probe_data.pt \
    --output_dir probe_results
```

### Step 4: Generate Heatmap
```bash
!python probing-llm-math/src/stratified_eval.py \
    --samples_per_cell 5 \
    --output_dir eval_results
```

---

## 5. Expected Outputs

### Probe Training Outputs
```
probe_results/
├── success_probe_accuracy.png     # 0%/50%/100% comparison
├── topic_probe_accuracy.png       # Layer sweep for topic
├── difficulty_probe_classification.png
├── difficulty_probe_regression.png
├── pca_layer20_correctness.png
├── pca_layer20_topic.png
├── pca_layer20_difficulty.png
└── summary.json
```

### Stratified Eval Outputs
```
eval_results/
├── accuracy_heatmap.png           # Topic × Difficulty heatmap
├── summary.txt                    # Text summary + cliff detection
├── eval_log.csv                   # Per-problem results
└── results.pt                     # Full results for re-analysis
```

---

## 6. File Structure

```
probing-llm-math/
├── src/
│   ├── model_utils.py          # Model loading & generation
│   ├── evaluator.py            # Answer verification (45 tests)
│   ├── collect_probe_data.py   # Data collection w/ metadata
│   ├── train_probe.py          # All 3 probe types
│   └── stratified_eval.py      # Heatmap generation
├── test_evaluator.py           # Evaluator test suite
├── pipeline_colab.ipynb        # Main notebook
├── current_state.md            # This file
├── data/
│   ├── problems.csv            # Stratified problem set
│   └── data_script.py          # Problem set generator
└── archive/                    # Old notebooks (reference only)
```

---

## 7. Technical Notes

### Hidden State Capture
- **0%**: Last token of prompt (question-only)
- **50%**: Token at `prompt_len + response_len/2`
- **100%**: Last token of full response

### Layer Sweep
- All layers are probed
- Best layer typically in range 15-25 for 7B models
- Results plotted for visual identification

### Evaluation Robustness
- Custom LaTeX-to-SymPy converter
- Symbolic comparison with numerical fallback (1e-9 tolerance)
- Element-wise matrix/tuple/set comparison

---

## 8. Next Steps (Optional Enhancements)

### If Time Permits:
1. **Multi-sample decoding**: Run 5 samples per problem for variance estimation
2. **Cross-validation**: K-fold CV instead of single train/test split
3. **Layer sweep optimization**: Every 4th layer first, then zoom in
4. **Intervention experiments**: Modify activations at probe directions

### Analysis Questions:
1. Does success probe accuracy improve from 0% → 50%?
2. Which layer best predicts topic? (Expect early-mid layers)
3. Do difficulty and success probes correlate?
4. Where are the biggest accuracy cliffs?

---

## 9. Colab Resource Estimates

| Task | GPU Time | VRAM |
|------|----------|------|
| Data collection (100 samples) | ~1-1.5 hrs | ~8GB |
| Data collection (200 samples) | ~2-3 hrs | ~8GB |
| Probe training | <5 min | CPU only |
| Stratified eval (100 problems) | ~2 hrs | ~8GB |

**Recommendation**: Use T4 for data collection, CPU runtime for probe training.

---

*Document updated: Nov 29, 2025*
*All core infrastructure complete and tested.*
