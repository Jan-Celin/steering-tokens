import torch
import re
from typing import Dict, List, Tuple, Optional


def parse_numeric_output(text: str) -> Optional[float]:
    """
    Extract numeric value from decoded text.
    Handles formats like "30.5", "-42", "1.23e-4", etc.
    Returns None if no valid number is found.
    """
    # Match floating point or integer numbers (including scientific notation)
    # TODO-JC: Check this regex!
    match = re.search(r'[+-]?\d+\.?\d*([eE][+-]?\d+)?', text.strip())
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def compute_mse_batch(
    model,
    batch: Dict,
    tokenizer,
    device: str,
    max_new_tokens: int = 16,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Evaluate model on a batch using greedy decoding.
    
    Args:
        model: The intervention model
        batch: Batch dict with keys: input_ids, attention_mask, left, right, operator, target_text
        tokenizer: Tokenizer for decoding
        device: Device to run on
        max_new_tokens: Max tokens to generate
    
    Returns:
        per_sample_mses: List of MSE values (one per sample)
        stats: Dict with keys like 'mean_mse', 'std_mse', 'exact_match_count', 'parse_error_count'
    """
    model.eval()
    
    per_sample_mses = []
    exact_matches = 0
    parse_errors = 0
    
    batch_size = batch["input_ids"].size(0)
    
    with torch.no_grad():
        for sample_idx in range(batch_size):
            # Get this sample's input
            input_ids = batch["input_ids"][sample_idx:sample_idx+1]
            attention_mask = batch["attention_mask"][sample_idx:sample_idx+1]
            
            # Greedy decoding
            generated_ids = input_ids.clone()
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs
                
                # Get last token logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                    dim=1
                )
                
                # Stop if we generated end-of-sequence
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
            
            # Decode prediction and ground truth
            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            target_text = batch["target_text"][sample_idx]
            
            # Parse numeric values
            pred_value = parse_numeric_output(pred_text)
            target_value = parse_numeric_output(target_text)
            
            # Compute MSE
            if pred_value is not None and target_value is not None:
                mse = (pred_value - target_value) ** 2
                per_sample_mses.append(mse)
                
                # Check for exact match (within floating point tolerance)
                if abs(pred_value - target_value) < 1e-5:
                    exact_matches += 1
            else:
                # If parsing fails, record error and max penalty
                parse_errors += 1
                per_sample_mses.append(1e6)  # Large penalty for unparseable output
    
    # Compute aggregate statistics
    mses = torch.tensor(per_sample_mses, dtype=torch.float32)
    stats = {
        "mean_mse": float(mses.mean().item()),
        "std_mse": float(mses.std().item()),
        "median_mse": float(mses.median().item()),
        "mse_values": per_sample_mses,
        "exact_match_rate": exact_matches / batch_size,
        "parse_error_count": parse_errors,
    }
    
    model.train()
    return per_sample_mses, stats


def evaluate_epoch(
    model,
    dataloader,
    device: str,
    tokenizer=None,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on entire dataloader.
    
    Args:
        model: The intervention model
        dataloader: Data loader (train or eval)
        device: Device to run on
        tokenizer: Tokenizer for decoding (required)
        max_samples: Maximum number of samples to evaluate (for speed)
    
    Returns:
        aggregated stats across all samples
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for evaluation")
    
    model.eval()
    
    all_mses = []
    total_exact_matches = 0
    total_samples = 0
    total_parse_errors = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            per_sample_mses, batch_stats = compute_mse_batch(model, batch, tokenizer, device)
            
            all_mses.extend(per_sample_mses)
            total_exact_matches += int(batch_stats["exact_match_rate"] * batch["input_ids"].size(0))
            total_samples += batch["input_ids"].size(0)
            total_parse_errors += batch_stats["parse_error_count"]
            
            if max_samples and total_samples >= max_samples:
                break
    
    mses = torch.tensor(all_mses, dtype=torch.float32)
    
    model.train()
    
    return {
        "mean_mse": float(mses.mean().item()),
        "std_mse": float(mses.std().item()),
        "median_mse": float(mses.median().item()),
        "mse_values": all_mses,
        "exact_match_rate": total_exact_matches / total_samples,
        "total_samples": total_samples,
        "parse_error_count": total_parse_errors,
    }
