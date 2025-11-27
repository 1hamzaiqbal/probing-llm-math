import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen2.5-Math-7B-Instruct", device="cuda"):
    """
    Loads the model and tokenizer.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_answer(model, tokenizer, question, device="cuda"):
    """
    Generates an answer for a math question and returns the text and hidden states.
    Enforces a 'direct answer' style via system prompt and generation config.
    """
    
    # Qwen-Math specific prompt format
    # It often responds better to "Please reason step by step, and put your final answer within \\boxed{}."
    # But since we want NO thinking, we try to force it.
    # However, Qwen-Math is heavily tuned for CoT. The best way to extract answer is to let it output \boxed{}.
    
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
            max_new_tokens=512,   # Allow some tokens in case it chatters, but we'll extract boxed
            do_sample=False,      # Greedy decoding
            # temperature=0.0,    # Invalid with do_sample=False
            top_p=1.0,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    
    # Extract text
    # The generated_ids.sequences contains prompt + new tokens
    # We only want the new tokens for the answer text
    new_tokens = generated_ids.sequences[0][len(model_inputs.input_ids[0]):]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Hidden states: generated_ids.hidden_states is a tuple (one per generated token)
    # Each element is a tuple of (one per layer) tensors.
    # We might want the last token's hidden state from the last layer, or all of them.
    # For now, let's return the full object or a simplified version.
    # Let's return the raw hidden_states tuple for the caller to process.
    
    return answer_text, generated_ids.hidden_states

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
