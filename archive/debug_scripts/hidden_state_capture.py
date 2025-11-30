import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", output_hidden_states=True
    ).eval()
    return tok, mdl

@torch.no_grad()
def question_only_states(model, tokenizer, question: str, max_len=1024):
    enc = tokenizer(question, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)  # no generation
    # last token (question end)
    last_idx = enc["input_ids"].shape[1] - 1
    # collect last-token residual/hidden per layer
    hs = [h[:, last_idx, :].float().cpu().numpy().squeeze(0) for h in out.hidden_states]
    # hs: list of [hidden_dim], length = n_layers+1 (incl. embeddings)
    return np.stack(hs)  # shape: (n_layers+1, hidden_dim)
