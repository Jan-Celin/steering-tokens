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

    total_loss = 0.0
    total_exact_matches = 0
    total_examples = 0

    per_example_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)

            # 1. Calculate teacher-forced loss
            if hasattr(model, "evaluation_step"):
                logits, labels, target_attention_mask = model.evaluation_step(batch)
            else:
                raise AttributeError(
                    "Model does not implement evaluation_step(batch)."
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

            # 2. Perform autoregressive generation for actual metrics
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=labels.size(1) + 5
            )

            for i in range(labels.size(0)):
                src = batch.get("source_text", [None] * labels.size(0))[i]
                prompt_len = int(attention_mask[i].sum().item())
                
                if tokenizer is not None:
                    tgt_ids = labels[i][mask[i]].tolist()
                    try:
                        tgt_text = tokenizer.decode(tgt_ids, skip_special_tokens=True).strip()
                        pred_ids = generated_ids[i][prompt_len:].tolist()
                        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                    except Exception:
                        tgt_text = ""
                        pred_text = ""
                else:
                    tgt_text = None
                    pred_text = None

                if tgt_text is not None and pred_text is not None and tgt_text.lower() == pred_text.lower():
                    total_exact_matches += 1

                per_example_results.append({
                    "source_text": src,
                    "target_text": tgt_text,
                    "predicted_text": pred_text,
                })
                
            total_examples += labels.size(0)

    mean_loss = total_loss / max(len(dataloader), 1)
    exact_match_rate = total_exact_matches / max(total_examples, 1)

    return {
        "mean_loss": mean_loss,
        "exact_match_rate": exact_match_rate,
        "examples": per_example_results,
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    device = config["training"]["device"] if torch.cuda.is_available() else "cpu"
    
    print("Evaluation logic configured.")
    pass

if __name__ == "__main__":
    main()
