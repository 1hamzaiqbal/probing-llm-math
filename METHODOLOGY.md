# Methodology: Probing Internal Signals in Math LLMs

## Overview

This project investigates what a math-specialized LLM "knows" about problems before, during, and after generating solutions. We train linear probes on frozen hidden state activations to decode:

1. **Success Prediction** - Can the model predict if it will answer correctly?
2. **Topic Classification** - Does the model encode the mathematical topic?
3. **Difficulty Estimation** - Does the model encode problem difficulty?

## Model

- **Primary Model**: `Qwen/Qwen2.5-Math-1.5B-Instruct`
  - 1.5B parameters, math-specialized
  - 29 transformer layers, 1536 hidden dimension
  - 4-bit quantization for efficient inference
  - ~50% accuracy on MATH dataset (good for balanced correct/incorrect samples)

- **Alternative**: `Qwen/Qwen2.5-Math-7B-Instruct`
  - Higher accuracy (~80%), but slower and harder to get balanced errors

## Dataset

**MATH Dataset** (Hendrycks et al.)
- Competition mathematics problems
- 4 topics used: Algebra, Number Theory, Counting & Probability, Precalculus
- 5 difficulty levels (Level 1-5)
- Geometry excluded (requires visual reasoning)

## Hidden State Extraction

### Checkpoints
We extract activations at three points during generation:

| Checkpoint | Description | What it captures |
|------------|-------------|------------------|
| **0%** | Last token of prompt, before any generation | Pre-answer "intuition" |
| **50%** | Token at midpoint of generated response | Mid-reasoning state |
| **100%** | Final token of complete response | Post-answer state |

### Pooling Methods

1. **Last Token** (position -1)
   - Extract hidden state at a single token position
   - Common in classification tasks

2. **Mean Pooling**
   - Average hidden states over all tokens in range
   - 0%: Mean over all prompt tokens
   - 50%: Mean over prompt + first half of response
   - 100%: Mean over all tokens

### Extraction Process

```python
# For each sample:
1. Build prompt with system message + question
2. Generate response with greedy decoding
3. Extract hidden states at all layers for each checkpoint
4. Store: [num_layers, hidden_dim] tensor per checkpoint
```

## Probe Training

### Linear Probes (Logistic Regression)

For binary classification (success prediction):
```python
LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
```

For multi-class (topic classification):
```python
LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
```

For regression (difficulty):
```python
Ridge(alpha=1.0)
```

### Difference-of-Means Probes

Inspired by "The Geometry of Truth" (Marks et al., 2023):

```python
# Compute mean activation for each class
mu_pos = mean(activations[correct_samples])
mu_neg = mean(activations[incorrect_samples])

# Direction vector
w = mu_pos - mu_neg
w = w / ||w||  # Normalize

# Predict by projecting onto direction
prediction = sign(dot(activation, w))
```

Advantages:
- No training required (closed-form)
- More interpretable
- Less prone to overfitting

### MLP Probes (Non-linear)

Two-layer neural network:
```python
MLP:
  Linear(hidden_dim, 128)
  ReLU()
  Dropout(0.3)
  Linear(128, num_classes)
```

Trained with:
- Adam optimizer, lr=0.001
- Cross-entropy loss
- 100 epochs, batch size 32
- Early stopping on validation loss

## Evaluation Metrics

### Success Probes
- **Accuracy**: Fraction of correct predictions
- **Baseline**: 50% (balanced dataset)

### Topic Probes
- **Accuracy**: Multi-class accuracy
- **Baseline**: ~63% (majority class - algebra)
- **Per-class F1**: Especially important for minority classes

### Difficulty Probes
- **Classification Accuracy**: 5-class
- **Baseline**: 20% (random guess)
- **R² Score**: For regression (how much variance explained)

## Control Tasks

### Shuffled Labels
Train probe on randomly shuffled labels to verify:
- Expected accuracy: ~50% (chance)
- If significantly higher: possible overfitting or data leakage

## Answer Evaluation

### Challenges
Mathematical answer comparison is non-trivial:
- LaTeX formatting differences: `\frac12` vs `\frac{1}{2}`
- Equivalent expressions: `-3(x+2)(x-1)` vs `-3x^2-3x+6`
- Variable prefixes: `x = 3` vs `3`
- Units: `575 students` vs `575`

### Solution
Multi-strategy evaluator:
1. String normalization (LaTeX cleanup)
2. Numeric comparison (float equality)
3. SymPy symbolic equivalence
4. Tuple/set element-wise comparison
5. Matrix element comparison

See `src/evaluator.py` for implementation (97 test cases).

## Experimental Setup

### Data Collection
```bash
python src/collect_probe_data.py \
    --num_samples 500 \
    --balance \                    # Equal correct/incorrect
    --pooling both \               # Last token + mean
    --seed 42
```

### Probe Training
```bash
python src/train_probe.py \
    --data_file probe_data.pt \
    --output_dir probe_results
```

### Stratified Evaluation
```bash
python src/stratified_eval.py \
    --samples_per_cell 20 \
    --output_dir eval_results
```

## Limitations

1. **Sample Size**: 500 samples may not capture all patterns
2. **Single Model**: Results may not generalize to other architectures
3. **Linear Probes**: May miss non-linear structure in representations
4. **Causal Claims**: Correlation ≠ causation; probes show what's decodable, not what's used
5. **Evaluation Edge Cases**: Some mathematical equivalences may be missed

## References

- Hendrycks, D. et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset.
- Marks, S. et al. (2023). The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations.
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances.

