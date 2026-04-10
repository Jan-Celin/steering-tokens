from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_id, device):
    """
    Loads the specified model and tokenizer from HuggingFace, ensuring the model is moved to the correct device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    # Load model; Using AutoModelForCausalLM for causal language modeling tasks
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)

    return model, tokenizer
