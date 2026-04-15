import torch
from tqdm import tqdm


def _move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        moved_batch[key] = value.to(device) if torch.is_tensor(value) else value
    return moved_batch


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for step_idx, batch in enumerate(pbar, start=1):
        optimizer.zero_grad()

        batch = _move_batch_to_device(batch, device)
        step_output = model.training_step(batch)
        loss = step_output["loss"] if isinstance(step_output, dict) else step_output

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / step_idx)

    return total_loss / len(dataloader)
