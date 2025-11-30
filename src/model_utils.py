import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", device="cuda", load_in_4bit=True):
    """
    Loads the model and tokenizer.
    Args:
        load_in_4bit (bool): If True, load in 4-bit quantization (recommended for T4 GPU).
    """
    print(f"Loading model: {model_name} (4-bit: {load_in_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model_kwargs = {
        "device_map": device,
        "trust_remote_code": True,
    }
    
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


def generate_answer(model, tokenizer, question, device="cuda"):
    """
    Generates an answer for a math question and returns the text, hidden states, and truncation status.
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
    
    max_new_tokens = 1024
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # Greedy decoding
            top_p=1.0,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    
    # Extract text
    new_tokens = generated_ids.sequences[0][len(model_inputs.input_ids[0]):]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Check for truncation
    is_truncated = len(new_tokens) == max_new_tokens
    
    return answer_text, generated_ids.hidden_states, is_truncated


def get_hidden_states_for_text(model, tokenizer, text, device="cuda"):
    """
    Helper to get hidden states for a full string (prompt + answer) via a single forward pass.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states
