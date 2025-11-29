import torch
from datasets import load_dataset
from tqdm import tqdm
import model_utils
import evaluator
import os

import argparse
import csv

import argparse
import csv
import random
from datasets import load_dataset, concatenate_datasets

def load_math_dataset(subjects=None, level=None):
    if subjects is None:
        subjects = ['algebra', 'number_theory', 'intermediate_algebra', 'precalculus']
    
    datasets = []
    for subject in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test", trust_remote_code=True)
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading subject {subject}: {e}")
            
    if not datasets:
        raise ValueError("Could not load any MATH datasets.")
        
    full_ds = concatenate_datasets(datasets)
    
    if level:
        # Filter by level (e.g., "Level 5")
        full_ds = full_ds.filter(lambda x: level in x['level'])
        
    return full_ds

def collect_data(num_samples=200, output_file="probe_data.pt", audit_file="probe_audit.csv", dataset_name="gsm8k", level=None, balance=False):
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif dataset_name.lower() == "math":
        dataset = load_math_dataset(level=level)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    model, tokenizer = model_utils.load_model(load_in_4bit=True)
    
    data = []
    
    # Prepare audit file
    audit_fields = ["question", "ground_truth", "prediction", "clean_prediction", "is_correct"]
    with open(audit_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
    
    print(f"Collecting data for {num_samples} samples...")
    if balance:
        print("Balanced mode enabled: Will try to collect equal number of correct and incorrect samples.")
        target_per_class = num_samples // 2
        correct_count = 0
        incorrect_count = 0
    
    print(f"Audit log will be saved to: {audit_file}")
    
    # Shuffle dataset indices to get random samples
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    pbar = tqdm(total=num_samples)
    
    for i in indices:
        if len(data) >= num_samples:
            break
            
        if balance:
            if correct_count >= target_per_class and incorrect_count >= target_per_class:
                break
        
        item = dataset[i]
        
        if dataset_name.lower() == "gsm8k":
            question = item['question']
            ground_truth = item['answer'].split("####")[-1].strip()
        else: # MATH
            question = item['problem']
            # Extract boxed answer from solution for cleaner comparison
            ground_truth = evaluator.extract_answer(item['solution'])
            
        # Generate
        pred_text, _, is_truncated = model_utils.generate_answer(model, tokenizer, question)
        
        # Evaluate
        clean_pred = evaluator.extract_answer(pred_text)
        is_correct = evaluator.is_equivalent(clean_pred, ground_truth)
        
        # Balance check
        if balance:
            if is_correct and correct_count >= target_per_class:
                continue # Skip if we have enough correct
            if not is_correct and incorrect_count >= target_per_class:
                continue # Skip if we have enough incorrect
                
        # Audit Log
        with open(audit_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=audit_fields)
            writer.writerow({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": pred_text,
                "clean_prediction": clean_pred,
                "is_correct": is_correct
            })
        
        # Construct full text for forward pass
        system_prompt = (
            "You are a math solver. "
            "Please provide the final answer directly. "
            "Put your final answer within \\boxed{}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        full_text = prompt_text + pred_text
        
        # Get hidden states for the full sequence
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_activations_prompt = []
        layer_activations_response = []
        
        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            # layer_hidden shape: (1, seq_len, hidden_dim)
            prompt_act = layer_hidden[0, prompt_len - 1, :].cpu()
            response_act = layer_hidden[0, -1, :].cpu()
            
            layer_activations_prompt.append(prompt_act)
            layer_activations_response.append(response_act)
            
        data.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred_text,
            "is_correct": is_correct,
            "prompt_activations": torch.stack(layer_activations_prompt), 
            "response_activations": torch.stack(layer_activations_response)
        })
        
        if is_correct:
            correct_count = correct_count + 1 if balance else 0
        else:
            incorrect_count = incorrect_count + 1 if balance else 0
            
        pbar.update(1)
        if balance:
            pbar.set_description(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
        
        # Periodic save
        if len(data) % 10 == 0:
            torch.save(data, output_file)

    pbar.close()
    torch.save(data, output_file)
    print(f"Finished! Saved {len(data)} samples to {output_file}")
    if balance:
        print(f"Final counts - Correct: {correct_count}, Incorrect: {incorrect_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hidden states for linear probing.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to collect.")
    parser.add_argument("--output_file", type=str, default="probe_data.pt", help="Output file for hidden states.")
    parser.add_argument("--audit_file", type=str, default="probe_audit.csv", help="Output file for audit log.")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"], help="Dataset to use.")
    parser.add_argument("--level", type=str, default=None, help="Difficulty level for MATH (e.g., 'Level 5').")
    parser.add_argument("--balance", action="store_true", help="Ensure balanced correct/incorrect samples.")
    
    args = parser.parse_args()
    
    collect_data(
        num_samples=args.num_samples, 
        output_file=args.output_file, 
        audit_file=args.audit_file,
        dataset_name=args.dataset,
        level=args.level,
        balance=args.balance
    )
