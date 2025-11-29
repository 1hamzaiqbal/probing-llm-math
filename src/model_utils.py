import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_name="Qwen/Qwen2.5-Math-7B-Instruct", device="cuda", load_in_4bit=True):
    """
    Loads the model and tokenizer with optimizations.
    
    Args:
        load_in_4bit (bool): If True, load in 4-bit quantization (recommended for T4 GPU).
    """
    print(f"Loading model: {model_name} (4-bit: {load_in_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model_kwargs = {
        "device_map": device,
        "trust_remote_code": True,
    }
    
    # Try to use flash attention for speedup (if available)
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Using Flash Attention 2")
    except Exception:
        pass
    
    if load_in_4bit:
        # Use proper BitsAndBytesConfig (avoids deprecation warning)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


def generate_answer(model, tokenizer, question, device="cuda", 
                    max_new_tokens=1024, capture_hidden_states=True):
    """
    Generates an answer for a math question.
    
    Args:
        max_new_tokens: Maximum tokens to generate (default 1024, reduced for speed)
        capture_hidden_states: If False, skip hidden state capture (2-3x faster)
    
    Returns:
        answer_text: Generated text
        hidden_states: Hidden states (or None if capture_hidden_states=False)
        is_truncated: Whether output was truncated
    """
    
    # Qwen-Math specific prompt format
    system_prompt = (
        "You are a math solver. "
        "Please provide the final answer directly. "
        "Put your final answer within \\boxed{}."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # Greedy decoding
            top_p=1.0,
            return_dict_in_generate=True,
            output_hidden_states=capture_hidden_states
        )
    
    # Extract text
    new_tokens = generated_ids.sequences[0][len(model_inputs.input_ids[0]):]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Check for truncation
    is_truncated = len(new_tokens) == max_new_tokens
    
    hidden_states = generated_ids.hidden_states if capture_hidden_states else None
    
    return answer_text, hidden_states, is_truncated


def generate_answer_fast(model, tokenizer, question, device="cuda", max_new_tokens=768):
    """
    Fast generation without hidden state capture.
    Use this for evaluation-only runs (stratified_eval.py).
    
    ~2-3x faster than generate_answer with hidden states.
    """
    system_prompt = (
        "You are a math solver. "
        "Please provide the final answer directly. "
        "Put your final answer within \\boxed{}."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    is_truncated = len(new_tokens) == max_new_tokens
    
    return answer_text, is_truncated


def get_hidden_states_for_text(model, tokenizer, text, device="cuda"):
    """
    Helper to get hidden states for a full string (prompt + answer) via a single forward pass.
    Useful if we want to probe the representation of the final token of the answer.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states is a tuple (one per layer)
    # Each tensor is (batch, seq_len, hidden_size)
    return outputs.hidden_states
