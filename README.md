# Probing Internal Signals in Math LLMs

This project investigates what a math-specialized LLM "knows" about problems before, during, and after generating solutions. We train linear probes on frozen hidden state activations to decode success prediction, topic classification, and difficulty estimation.

## Key Findings

- **Success Prediction**: 75-80% accuracy at 0% checkpoint - the model "knows" it will fail before answering
- **Mean Pooling**: Outperforms last-token extraction by 2-5%
- **Topic Classification**: 86% accuracy (expected - distinct vocabulary)
- **Difficulty Estimation**: 46% (weak signal)

## Quick Start (Colab)

```python
# Clone and setup
!git clone -b colab-pipeline-setup https://github.com/1hamzaiqbal/probing-llm-math.git
!pip install -r probing-llm-math/requirements.txt

# Collect data (500 balanced samples, ~7 hours)
!python probing-llm-math/src/collect_probe_data.py \
    --num_samples 500 --balance --pooling both

# Train probes
!python probing-llm-math/src/train_probe.py \
    --data_file probe_data.pt --output_dir probe_results --mlp

# Analyze confidence calibration
!python probing-llm-math/src/confidence_analysis.py \
    --data_file probe_data.pt --output_dir confidence_analysis
```

## Project Structure

```
probing-llm-math/
├── src/
│   ├── collect_probe_data.py   # Data collection with pooling options
│   ├── train_probe.py          # Linear + MLP probe training
│   ├── evaluator.py            # Robust answer comparison (97 tests)
│   ├── model_utils.py          # Model loading (1.5B default)
│   ├── stratified_eval.py      # Accuracy heatmap generation
│   ├── analyze_tokens.py       # Token count analysis
│   └── confidence_analysis.py  # Calibration & progression
├── test_evaluator.py           # Evaluator test suite
├── pipeline_colab.ipynb        # Main Colab notebook
├── METHODOLOGY.md              # Full methodology writeup
├── RESULTS_TEMPLATE.md         # Results template
├── PROGRESS.md                 # Current status
├── proposal/                   # Original proposal
└── archive/                    # Old/superseded files
```

## Model

- **Default**: `Qwen/Qwen2.5-Math-1.5B-Instruct` (faster iteration)
- **Alternative**: `Qwen/Qwen2.5-Math-7B-Instruct` (higher accuracy)
- 4-bit quantization via bitsandbytes

## Documentation

- [METHODOLOGY.md](METHODOLOGY.md) - Full methodology with references
- [PROGRESS.md](PROGRESS.md) - Current project status
- [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md) - Template for results

## Requirements

```
torch>=2.0
transformers>=4.35
bitsandbytes
datasets
scikit-learn
sympy
matplotlib
seaborn
tqdm
```
