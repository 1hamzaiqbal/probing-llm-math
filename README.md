# Math LLM Probing & Evaluation Pipeline

This repository contains a pipeline to evaluate `Qwen/Qwen2.5-Math-7B-Instruct` on math problems, enforcing direct answers and extracting hidden states.

## Setup

1.  **Google Colab**:
    *   Upload the `src` folder and `pipeline_colab.ipynb` to your Colab environment.
    *   Open `pipeline_colab.ipynb`.
    *   Run the cells. The first cell installs necessary dependencies.

2.  **Local (with GPU)**:
    *   Install dependencies: `pip install -r requirements.txt`
    *   Run the notebook or use the scripts in `src/`.

## Structure

*   `src/model_utils.py`: Functions to load the model and generate answers with hidden states.
*   `src/evaluator.py`: Functions to extract answers and verify correctness using Regex and SymPy.
*   `pipeline_colab.ipynb`: The main notebook to run the experiment.
*   `cleanup_script.py`: Script used to archive old files (already run).
*   `archive/`: Contains old/duplicate files.

## Key Features

*   **Model**: Uses `Qwen/Qwen2.5-Math-7B-Instruct`.
*   **Direct Answers**: Enforced via system prompt and `temperature=0`.
*   **Evaluation**: Lightweight Regex + SymPy check (no heavy judge model needed).
