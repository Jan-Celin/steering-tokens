from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model; Using AutoModelForCausalLM for causal language modeling tasks
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    return model, tokenizer
