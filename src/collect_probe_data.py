import torch
from datasets import load_dataset
from tqdm import tqdm
import model_utils
import evaluator
import os

import argparse
import csv

def collect_data(num_samples=200, output_file="probe_data.pt", audit_file="probe_audit.csv"):
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Load model
    model, tokenizer = model_utils.load_model(load_in_4bit=True)
    
    data = []
    
    # Prepare audit file
    audit_fields = ["question", "ground_truth", "prediction", "clean_prediction", "is_correct"]
    with open(audit_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
    
    print(f"Collecting data for {num_samples} samples...")
    print(f"Audit log will be saved to: {audit_file}")
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        item = dataset[i]
        question = item['question']
        ground_truth = item['answer'].split("####")[-1].strip()
        
        # Generate
        pred_text, _, is_truncated = model_utils.generate_answer(model, tokenizer, question)
        
        # Evaluate
        clean_pred = evaluator.extract_answer(pred_text)
        is_correct = evaluator.is_equivalent(clean_pred, ground_truth)
        
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
        
        # Periodic save
        if (i + 1) % 10 == 0:
            torch.save(data, output_file)
            print(f"Saved {i+1} samples to {output_file}")

    torch.save(data, output_file)
    print(f"Finished! Saved {len(data)} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hidden states for linear probing.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to collect.")
    parser.add_argument("--output_file", type=str, default="probe_data.pt", help="Output file for hidden states.")
    parser.add_argument("--audit_file", type=str, default="probe_audit.csv", help="Output file for audit log.")
    
    args = parser.parse_args()
    
    collect_data(num_samples=args.num_samples, output_file=args.output_file, audit_file=args.audit_file)
