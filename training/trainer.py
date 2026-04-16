import torch
from tqdm import tqdm
from typing import Callable, Optional, Dict, Any

MAX_STEPS = None

def _move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        moved_batch[key] = value.to(device) if torch.is_tensor(value) else value
    return moved_batch


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    eval_fn: Optional[Callable] = None,
    eval_dataloader: Optional[Any] = None,
    eval_kwargs: Optional[Dict] = None,
):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        eval_fn: Optional evaluation function to call after epoch
        eval_dataloader: Optional dataloader for evaluation
        eval_kwargs: Optional kwargs to pass to eval_fn (e.g., tokenizer)
    
    Returns:
        (epoch_loss, losses, eval_results): Average loss, per-step losses, and eval metrics
    """
    model.train()
    total_loss = 0
    losses = []

    pbar = tqdm(dataloader, desc="Training")
    for step_idx, batch in enumerate(pbar, start=1):
        if MAX_STEPS and step_idx > MAX_STEPS:
            break

        optimizer.zero_grad()

        batch = _move_batch_to_device(batch, device)
        step_output = model.training_step(batch)
        loss = step_output["loss"] if isinstance(step_output, dict) else step_output

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

        pbar.set_postfix(loss=total_loss / step_idx)

    epoch_loss = total_loss / len(dataloader)
    
    # Run evaluation if provided
    eval_results = None
    if eval_fn and eval_dataloader:
        if eval_kwargs is None:
            eval_kwargs = {}
        eval_results = eval_fn(model, eval_dataloader, device, **eval_kwargs)
    
    return epoch_loss, losses, eval_results
