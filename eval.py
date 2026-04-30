import argparse
from utils.config import load_config
from models.model_manager import load_model_and_tokenizer
import torch
def _move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        moved_batch[key] = value.to(device) if torch.is_tensor(value) else value
    return moved_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_tokens.yaml")
    return parser.parse_args()

def evaluate(model, dataloader, device, tokenizer=None):
    model.eval()

    if tokenizer is None and hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer

    pad_token_id = None if tokenizer is None else tokenizer.pad_token_id

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_exact_matches = 0
    total_examples = 0

    per_example_results = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)

            if hasattr(model, "evaluation_step"):
                logits, labels, target_attention_mask = model.evaluation_step(batch)
            else:
                raise AttributeError(
                    "Model does not implement evaluation_step(batch), which is required for evaluation."
                )

            mask = target_attention_mask.bool()
            loss_labels = labels.clone()
            loss_labels[~mask] = -100

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                loss_labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            total_correct += ((predictions == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            per_example_match = ((predictions == labels) | ~mask).all(dim=1)
            total_exact_matches += per_example_match.sum().item()
            total_examples += labels.size(0)

            # collect per-example decoded texts if tokenizer available
            if tokenizer is not None:
                for i in range(labels.size(0)):
                    src = batch.get("source_text", [None] * labels.size(0))[i]
                    tgt_ids = labels[i][mask[i]].tolist()
                    pred_ids = predictions[i][mask[i]].tolist()
                    try:
                        tgt_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
                        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    except Exception:
                        tgt_text = ""
                        pred_text = ""
                    per_example_results.append({
                        "source_text": src,
                        "target_text": tgt_text,
                        "predicted_text": pred_text,
                    })
            else:
                for i in range(labels.size(0)):
                    src = batch.get("source_text", [None] * labels.size(0))[i]
                    per_example_results.append({
                        "source_text": src,
                        "target_text": None,
                        "predicted_text": None,
                    })

    mean_loss = total_loss / max(len(dataloader), 1)
    token_accuracy = total_correct / max(total_tokens, 1)
    exact_match_rate = total_exact_matches / max(total_examples, 1)

    return {
        "mean_loss": mean_loss,
        "token_accuracy": token_accuracy,
        "exact_match_rate": exact_match_rate,
        "examples": per_example_results,
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    device = config["training"]["device"] if torch.cuda.is_available() else "cpu"
    
    # Place evaluation setup here
    pass

if __name__ == "__main__":
    main()
