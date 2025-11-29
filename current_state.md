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

### 2.2 Core Modules (`src/`)

| File | Status | Purpose |
|------|--------|---------|
| `model_utils.py` | ✅ Working | Load model (4-bit), generate with hidden states |
| `evaluator.py` | ✅ Working | Extract boxed answers, verify via `math-verify` |
| `collect_probe_data.py` | ✅ Working | Collect (question, hidden_states, is_correct) tuples with balancing |
| `train_probe.py` | ✅ Working | Train logistic regression probes per layer, generate plots |
| `hidden_state_capture.py` | ⚠️ Alternate | Simpler forward-pass capture (question-only states) |
| `topic_probe.py` | ⚠️ Skeleton | Helper functions, not integrated |

### 2.3 Data Sources

| Source | Script/Location | Status |
|--------|-----------------|--------|
| MATH dataset (HF) | `EleutherAI/hendrycks_math` | ✅ Used in `collect_probe_data.py` |
| GSM8K | HF `gsm8k/main` | ✅ Alternate option |
| Custom stratified | `data/problems.csv` (75 problems) | ✅ Created via `data/data_script.py` |

---

## 3. What Works Right Now

From `pipeline_colab.ipynb` outputs:

### ✅ Evaluation Pipeline
- Model loads successfully (4-bit)
- Generation with `\boxed{}` format works
- `math-verify` correctly handles: matrices, intervals, coordinates, surds, integrals
- Hidden states captured during generation

### ✅ Data Collection (`collect_probe_data.py`)
- Balanced sampling (equal correct/incorrect)
- Captures activations at:
  - **Prompt end** (last token of question) — `prompt_activations`
  - **Response end** (last token of answer) — `response_activations`
- Saves to `probe_data.pt` with audit CSV

### ✅ Probe Training (`train_probe.py`)
- Per-layer logistic regression
- Accuracy plot across all layers
- PCA visualization at representative layer

---

## 4. What's Missing / Incomplete

### 🔴 Critical Gaps

| Gap | Impact | Notes |
|-----|--------|-------|
| **No 50% checkpoint capture** | O3 (Success probe) incomplete | Current only captures 0% and 100%, not mid-generation |
| **No topic labels** | O2 (Topic probe) not trainable | `collect_probe_data.py` doesn't store topic |
| **No difficulty labels** | O4 (Difficulty probe) not trainable | `collect_probe_data.py` doesn't store level |
| **No stratified eval loop** | Can't generate heatmaps | Need topic × difficulty matrix |

### 🟡 Integration Issues

| Issue | Location | Fix Needed |
|-------|----------|------------|
| `topic_probe.py` unused | `src/` | Integrate with data collection |
| `hidden_state_capture.py` duplicates logic | `src/` | Consolidate or remove |
| `data/problems.csv` not used | `data/` | Wire up for stratified eval |
| Archive notebooks untested | `archive/` | Potentially useful for heatmap code |

### 🟢 Nice-to-Haves (Later)

- Layer sweep optimization (every 4th → zoom in)
- Multi-sample decoding (5 samples/problem for variance)
- Geometry exclusion filter

---

## 5. Proposed Data Schema

To support all three probes, captured data should include:

```python
{
    "problem_id": str,
    "topic": str,           # algebra | number_theory | counting_probability
    "difficulty": int,      # 1-5
    "question": str,
    "ground_truth": str,
    "prediction": str,
    "is_correct": bool,
    
    # Activations (all are tensors: [num_layers+1, hidden_dim])
    "activations_0pct": Tensor,    # After question, before generation
    "activations_50pct": Tensor,   # After generating ~50% of tokens
    "activations_100pct": Tensor,  # After full response
}
```

---

## 6. Immediate Next Steps

### Phase A: Complete Data Collection (Priority)

**Goal**: Extend `collect_probe_data.py` to capture all needed metadata and checkpoints.

```diff
# In collect_data():
+ store topic from dataset item
+ store difficulty (level) from dataset item
+ add 50% checkpoint capture (replay with truncated generation)
```

**Checkpoint capture strategy** (from proposal):
1. Generate full answer once → get `prediction`
2. Replay with `prompt + first_half_of_prediction` to get `activations_50pct`
3. This avoids streaming complexity

### Phase B: Stratified Evaluation Loop

**Goal**: Run model on stratified `problems.csv` → produce accuracy matrix.

```python
# Pseudocode
results = defaultdict(list)
for problem in problems_csv:
    pred = generate_answer(problem.question)
    correct = is_equivalent(pred, problem.answer)
    results[(problem.topic, problem.difficulty)].append(correct)

# Build heatmap
accuracy_matrix = compute_accuracy_per_cell(results)
plot_heatmap(accuracy_matrix)
```

### Phase C: Train All Three Probes

| Probe | X (features) | y (labels) | Notes |
|-------|--------------|------------|-------|
| Topic | `activations_0pct` | `topic` | 3-class classification |
| Success (0%) | `activations_0pct` | `is_correct` | Binary |
| Success (50%) | `activations_50pct` | `is_correct` | Compare to 0% |
| Difficulty | `activations_0pct` | `difficulty` | 5-class or regression |

---

## 7. Technical Decisions

### Hidden State Strategy
- **Use final token of each segment** (not mean-pooled)
  - More standard in probing literature
  - Captures "accumulated" representation
- **Every layer** initially, then sweep

### Evaluation
- `math-verify` is working well — keep it
- Fallback chain: string match → float parse → math-verify

### Quantization
- 4-bit is fine for inference accuracy
- Hidden states still have full precision in the activations

---

## 8. File Cleanup Recommendations

```
KEEP (core):
├── src/
│   ├── model_utils.py          # ✓
│   ├── evaluator.py            # ✓
│   ├── collect_probe_data.py   # ✓ (extend)
│   └── train_probe.py          # ✓ (extend)
├── pipeline_colab.ipynb        # ✓ main entrypoint
├── data/
│   ├── problems.csv            # ✓ stratified eval set
│   └── data_script.py          # ✓ generation script

CONSOLIDATE/REMOVE:
├── src/
│   ├── hidden_state_capture.py # merge into model_utils
│   └── topic_probe.py          # merge into train_probe
├── llm_math_probe.ipynb        # old, archive it
├── debug_eval.py               # debug, remove when done
├── reproduce_issue.py          # debug, remove when done
├── inspect_math_dataset.py     # debug, remove when done
```

---

## 9. Colab Resource Plan

| Phase | GPU Time | VRAM | Notes |
|-------|----------|------|-------|
| Data collection (200 samples) | ~2-3 hrs | ~8GB | 4-bit, T4 is fine |
| Probe training | <5 min | CPU only | Just sklearn |
| Full eval (150 problems × 5 samples) | ~5-6 hrs | ~8GB | Consider A100 for speed |

---

## 10. Key Code Pointers

### Model Loading
```python
# src/model_utils.py:4-25
model, tokenizer = load_model(model_name="Qwen/Qwen2.5-Math-7B-Instruct", load_in_4bit=True)
```

### Hidden State Capture (Generation)
```python
# src/model_utils.py:27-73
answer_text, hidden_states, is_truncated = generate_answer(model, tokenizer, question)
# hidden_states is a tuple of (layer_hidden_states for each generation step)
```

### Hidden State Capture (Forward Pass)
```python
# src/model_utils.py:75-86
hidden_states = get_hidden_states_for_text(model, tokenizer, full_text)
# Returns tuple: (layer0, layer1, ..., layerN), each (1, seq_len, hidden_dim)
```

### Answer Verification
```python
# src/evaluator.py:27-69
is_correct = is_equivalent(model_answer, ground_truth)
# Chain: string match → float compare → math-verify
```

---

## 11. Open Questions

1. **50% token boundary**: Use token count or character count? (Token is cleaner)
2. **Geometry handling**: Filter out or flag? (Proposal says exclude)
3. **Multi-sample decoding**: Do we need it for probing, or just for accuracy measurement?
4. **Layer sweep strategy**: Every 4th layer first, then zoom into promising regions?

---

## 12. Evaluation Issues — RESOLVED ✅

### Fixed Issues (Nov 29, 2025)

The evaluator has been significantly improved with SymPy-based symbolic comparison:

| Issue | Example | Status |
|-------|---------|--------|
| Fraction notation | `2/5` vs `\frac{2}{5}` | ✅ Fixed |
| Rationalized denominators | `1/\sqrt{2}` vs `\sqrt{2}/2` | ✅ Fixed |
| Complex number ordering | `i/5` vs `(1/5)i` | ✅ Fixed |
| Matrix element comparison | Element-wise symbolic | ✅ Fixed |
| Implicit multiplication | `2\sqrt{3}` | ✅ Fixed |
| Commutative addition | `1 + \sqrt{2}` vs `\sqrt{2} + 1` | ✅ Fixed |

### New Evaluator Features

- **Custom LaTeX-to-SymPy converter** for complex expressions
- **Matrix element-wise comparison** with symbolic evaluation
- **Numerical fallback** for approximate equality (1e-9 tolerance)
- **Debug helper** `debug_comparison(model_ans, gt)` for troubleshooting

### Test Coverage

Run `python test_evaluator.py` to validate. Current: **26/26 tests passing**.

### Remaining Validation Checklist

- [ ] Test on 20 problems from each topic to verify answer extraction
- [ ] Confirm hidden state tensor shapes are consistent across problems
- [ ] Verify balanced sampling actually produces 50/50 split
- [ ] Test on geometry problems (if we decide to include them)

---

## 13. Quick Start Commands

```bash
# In Colab, after running pipeline_colab.ipynb cells 1-4:

# Collect probe data with metadata
!python probing-llm-math/src/collect_probe_data.py \
    --num_samples 100 \
    --dataset math \
    --balance \
    --output_file probe_data_v2.pt \
    --audit_file probe_audit_v2.csv

# Train probes
!python probing-llm-math/src/train_probe.py \
    --data_file probe_data_v2.pt \
    --output_dir probe_results_v2
```

---

## 14. Next Session Agenda

1. **Validate evaluation** on edge cases (section 12)
2. **Extend `collect_probe_data.py`** with topic/difficulty/50% checkpoints
3. **Run stratified eval** to generate heatmap
4. **Train all three probes** and analyze layer-wise results

---

*Document generated from codebase analysis on Nov 29, 2025*

