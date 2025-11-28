import torch
from datasets import load_dataset
from tqdm import tqdm
import model_utils
import evaluator
import os

def collect_data(num_samples=200, output_file="probe_data.pt"):
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Load model
    model, tokenizer = model_utils.load_model(load_in_4bit=True)
    
    data = []
    
    print(f"Collecting data for {num_samples} samples...")
    for i in tqdm(range(min(num_samples, len(dataset)))):
        item = dataset[i]
        question = item['question']
        # GSM8K answers are often ".... #### 1234". We need to parse the number.
        # But math-verify might handle the full string if we pass it as gold?
        # Actually, math-verify expects the gold to be the final answer usually.
        # Let's extract the part after #### for gold, or use the full string if math-verify supports it.
        # Standard practice for GSM8K is to take the part after ####.
        ground_truth = item['answer'].split("####")[-1].strip()
        
        # Generate
        # We need the full hidden states to extract specific tokens
        # generate_answer returns (text, hidden_states, is_truncated)
        # hidden_states is a tuple of tuples (generated_tokens, layers)
        # Wait, generate_answer returns hidden_states for the *generated* tokens.
        # We also want the prompt's last token.
        # So we might need to do a separate forward pass on the prompt, 
        # OR we can get the prompt's hidden states if we modify generate_answer to return them,
        # OR we can just run a forward pass on the full sequence (prompt + completion) afterwards.
        # Running a forward pass on the full sequence is cleaner and easier to manage.
        
        pred_text, _, is_truncated = model_utils.generate_answer(model, tokenizer, question)
        
        # Evaluate
        clean_pred = evaluator.extract_answer(pred_text)
        is_correct = evaluator.is_equivalent(clean_pred, ground_truth)
        
        # Construct full text for forward pass
        # We need to replicate exactly what was fed to the model + what was generated.
        # model_utils.generate_answer uses a chat template.
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
        # We need to know the index of the last prompt token and the last generated token.
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # outputs.hidden_states is a tuple of (batch, seq_len, hidden_dim) tensors, one per layer.
        # We want to extract:
        # 1. Last token of prompt: index = prompt_len - 1
        # 2. Last token of response: index = -1
        
        layer_activations_prompt = []
        layer_activations_response = []
        
        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            # layer_hidden shape: (1, seq_len, hidden_dim)
            # Move to CPU to save VRAM
            prompt_act = layer_hidden[0, prompt_len - 1, :].cpu()
            response_act = layer_hidden[0, -1, :].cpu()
            
            layer_activations_prompt.append(prompt_act)
            layer_activations_response.append(response_act)
            
        data.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred_text,
            "is_correct": is_correct,
            "prompt_activations": torch.stack(layer_activations_prompt), # Shape: (num_layers, hidden_dim)
            "response_activations": torch.stack(layer_activations_response) # Shape: (num_layers, hidden_dim)
        })
        
        # Periodic save
        if (i + 1) % 10 == 0:
            torch.save(data, output_file)
            print(f"Saved {i+1} samples to {output_file}")

    torch.save(data, output_file)
    print(f"Finished! Saved {len(data)} samples to {output_file}")

if __name__ == "__main__":
    collect_data()
