"""
Collect hidden states for linear probing experiments.

Captures:
- Topic labels (algebra, number_theory, etc.)
- Difficulty levels (1-5)
- Activations at 0% (question-only), 50% (mid-generation), 100% (full response)
- Correctness labels
"""

import torch
import argparse
import csv
import random
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

import model_utils
import evaluator


# Mapping from dataset subject names to cleaner topic names
SUBJECT_TO_TOPIC = {
    'algebra': 'algebra',
    'intermediate_algebra': 'algebra',
    'number_theory': 'number_theory',
    'counting_and_probability': 'counting_probability',
    'prealgebra': 'algebra',
    'precalculus': 'precalculus',
    'geometry': 'geometry',  # We may filter this out
}


def load_math_dataset_with_metadata(subjects=None, level=None, exclude_geometry=True):
    """
    Load MATH dataset and preserve topic/level metadata.
    
    Returns a list of dicts with: problem, solution, topic, level, level_num
    """
    if subjects is None:
        subjects = ['algebra', 'number_theory', 'intermediate_algebra', 
                    'counting_and_probability', 'precalculus', 'prealgebra']
    
    if exclude_geometry and 'geometry' in subjects:
        subjects.remove('geometry')
    
    all_items = []
    
    for subject in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
            for item in ds:
                # Parse level number from string like "Level 3"
                level_str = item.get('level', 'Level 1')
                level_num = int(level_str.replace('Level ', ''))
                
                # Map subject to topic
                topic = SUBJECT_TO_TOPIC.get(subject, subject)
                
                all_items.append({
                    'problem': item['problem'],
                    'solution': item['solution'],
                    'topic': topic,
                    'level': level_str,
                    'level_num': level_num,
                    'subject': subject,  # Keep original subject for reference
                })
        except Exception as e:
            print(f"Error loading subject {subject}: {e}")
    
    if not all_items:
        raise ValueError("Could not load any MATH datasets.")
    
    # Filter by level if specified
    if level:
        if isinstance(level, int):
            all_items = [x for x in all_items if x['level_num'] == level]
        else:
            all_items = [x for x in all_items if level in x['level']]
    
    print(f"Loaded {len(all_items)} problems")
    print(f"Topics: {set(x['topic'] for x in all_items)}")
    print(f"Levels: {set(x['level_num'] for x in all_items)}")
    
    return all_items


def get_prompt_text(tokenizer, question):
    """Build the full prompt text for a question."""
    system_prompt = (
        "You are a math solver. "
        "Please provide the final answer directly. "
        "Put your final answer within \\boxed{}."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def get_activations_at_position(model, tokenizer, text, position=-1):
    """
    Get hidden state activations at a specific token position.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        text: Full text to process
        position: Token position to extract (-1 for last token)
    
    Returns:
        Tensor of shape [num_layers+1, hidden_dim]
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states is a tuple of (num_layers+1) tensors
    # Each tensor is (batch=1, seq_len, hidden_dim)
    activations = []
    for layer_hidden in outputs.hidden_states:
        act = layer_hidden[0, position, :].cpu()
        activations.append(act)
    
    return torch.stack(activations)  # [num_layers+1, hidden_dim]


def collect_data(
    num_samples=200,
    output_file="probe_data.pt",
    audit_file="probe_audit.csv",
    dataset_name="math",
    level=None,
    balance=False,
    capture_50pct=True,
    exclude_geometry=True,
):
    """
    Collect probe training data with full metadata.
    
    Captures activations at:
    - 0%: After question, before any generation (prompt_activations)
    - 50%: After generating half of the response tokens (mid_activations)
    - 100%: After full response (response_activations)
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "gsm8k":
        # GSM8K doesn't have topic/difficulty, so we use placeholders
        raw_dataset = load_dataset("gsm8k", "main", split="test")
        dataset = []
        for item in raw_dataset:
            dataset.append({
                'problem': item['question'],
                'solution': item['answer'],
                'topic': 'arithmetic',  # GSM8K is primarily arithmetic
                'level_num': 0,  # Unknown difficulty
                'level': 'Unknown',
            })
    elif dataset_name.lower() == "math":
        dataset = load_math_dataset_with_metadata(
            level=level, 
            exclude_geometry=exclude_geometry
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    model, tokenizer = model_utils.load_model(load_in_4bit=True)
    
    data = []
    
    # Prepare audit file with extended fields
    audit_fields = [
        "idx", "topic", "level", "question", "ground_truth", 
        "prediction", "clean_prediction", "is_correct", "num_tokens"
    ]
    with open(audit_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
    
    print(f"Collecting data for {num_samples} samples...")
    if balance:
        print("Balanced mode: collecting equal correct/incorrect samples.")
        target_per_class = num_samples // 2
        correct_count = 0
        incorrect_count = 0
    else:
        correct_count = 0
        incorrect_count = 0
    
    print(f"Audit log: {audit_file}")
    print(f"50% checkpoint capture: {'enabled' if capture_50pct else 'disabled'}")
    
    # Shuffle dataset indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    pbar = tqdm(total=num_samples)
    
    for idx in indices:
        if len(data) >= num_samples:
            break
            
        if balance:
            if correct_count >= target_per_class and incorrect_count >= target_per_class:
                break
        
        item = dataset[idx]
        question = item['problem']
        topic = item['topic']
        level_num = item['level_num']
        level_str = item['level']
        
        # Extract ground truth
        if dataset_name.lower() == "gsm8k":
            ground_truth = item['solution'].split("####")[-1].strip()
        else:
            ground_truth = evaluator.extract_answer(item['solution'])
        
        # Generate answer
        pred_text, _, is_truncated = model_utils.generate_answer(model, tokenizer, question)
        
        # Evaluate
        clean_pred = evaluator.extract_answer(pred_text)
        is_correct = evaluator.is_equivalent(clean_pred, ground_truth)
        
        # Balance check
        if balance:
            if is_correct and correct_count >= target_per_class:
                continue
            if not is_correct and incorrect_count >= target_per_class:
                continue
        
        # Build texts for activation capture
        prompt_text = get_prompt_text(tokenizer, question)
        full_text = prompt_text + pred_text
        
        # Tokenize to get lengths
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt")
        full_tokens = tokenizer(full_text, return_tensors="pt")
        prompt_len = prompt_tokens.input_ids.shape[1]
        full_len = full_tokens.input_ids.shape[1]
        response_len = full_len - prompt_len
        
        # === Capture 0% activations (last token of prompt, before generation) ===
        activations_0pct = get_activations_at_position(
            model, tokenizer, prompt_text, position=-1
        )
        
        # === Capture 100% activations (last token of full response) ===
        activations_100pct = get_activations_at_position(
            model, tokenizer, full_text, position=-1
        )
        
        # === Capture 50% activations (mid-response, via replay) ===
        activations_50pct = None
        if capture_50pct and response_len > 2:
            # Calculate 50% token position
            mid_response_tokens = response_len // 2
            mid_position = prompt_len + mid_response_tokens - 1  # -1 for 0-indexing
            
            # Get activations at mid position
            activations_50pct = get_activations_at_position(
                model, tokenizer, full_text, position=mid_position
            )
        
        # Write audit log
        with open(audit_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=audit_fields)
            writer.writerow({
                "idx": len(data),
                "topic": topic,
                "level": level_str,
                "question": question[:200] + "..." if len(question) > 200 else question,
                "ground_truth": ground_truth,
                "prediction": pred_text[:200] + "..." if len(pred_text) > 200 else pred_text,
                "clean_prediction": clean_pred,
                "is_correct": is_correct,
                "num_tokens": response_len,
            })
        
        # Build data record
        record = {
            "problem_id": f"{dataset_name}_{idx}",
            "topic": topic,
            "level_num": level_num,
            "level_str": level_str,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred_text,
            "is_correct": is_correct,
            "num_response_tokens": response_len,
            "activations_0pct": activations_0pct,    # [num_layers+1, hidden_dim]
            "activations_100pct": activations_100pct,  # [num_layers+1, hidden_dim]
        }
        
        if activations_50pct is not None:
            record["activations_50pct"] = activations_50pct
        
        # Legacy field names for compatibility with existing train_probe.py
        record["prompt_activations"] = activations_0pct
        record["response_activations"] = activations_100pct
        
        data.append(record)
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
            
        pbar.update(1)
        pbar.set_description(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
        
        # Periodic save
        if len(data) % 10 == 0:
            torch.save(data, output_file)

    pbar.close()
    torch.save(data, output_file)
    
    print(f"\n{'='*50}")
    print(f"Finished! Saved {len(data)} samples to {output_file}")
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
    
    # Print topic/level distribution
    topic_counts = {}
    level_counts = {}
    for d in data:
        topic_counts[d['topic']] = topic_counts.get(d['topic'], 0) + 1
        level_counts[d['level_num']] = level_counts.get(d['level_num'], 0) + 1
    
    print(f"\nTopic distribution: {topic_counts}")
    print(f"Level distribution: {level_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hidden states for linear probing.")
    parser.add_argument("--num_samples", type=int, default=200, 
                        help="Number of samples to collect.")
    parser.add_argument("--output_file", type=str, default="probe_data.pt", 
                        help="Output file for hidden states.")
    parser.add_argument("--audit_file", type=str, default="probe_audit.csv", 
                        help="Output file for audit log.")
    parser.add_argument("--dataset", type=str, default="math", choices=["gsm8k", "math"], 
                        help="Dataset to use.")
    parser.add_argument("--level", type=str, default=None, 
                        help="Difficulty level filter (e.g., '3' or 'Level 3').")
    parser.add_argument("--balance", action="store_true", 
                        help="Ensure balanced correct/incorrect samples.")
    parser.add_argument("--no_50pct", action="store_true",
                        help="Disable 50%% checkpoint capture (faster).")
    parser.add_argument("--include_geometry", action="store_true",
                        help="Include geometry problems (excluded by default).")
    
    args = parser.parse_args()
    
    # Parse level argument
    level = None
    if args.level:
        try:
            level = int(args.level)
        except ValueError:
            level = args.level  # Keep as string like "Level 3"
    
    collect_data(
        num_samples=args.num_samples, 
        output_file=args.output_file, 
        audit_file=args.audit_file,
        dataset_name=args.dataset,
        level=level,
        balance=args.balance,
        capture_50pct=not args.no_50pct,
        exclude_geometry=not args.include_geometry,
    )
