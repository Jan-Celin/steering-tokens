import argparse
from importlib import import_module
import torch
from utils.config import load_config
from models.model_manager import load_model_and_tokenizer
from interventions import get_intervention
from training.trainer import train_epoch
from eval import evaluate
from utils.visualization import plot_training_curves, save_eval_examples
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/translation.yaml")
    return parser.parse_args()

def print_epoch_summary(epoch, loss, eval_results):
    print(f"Epoch {epoch} - Loss: {loss:.4f}")
    if eval_results is not None:
        print("Evaluation Results:")
        for metric, value in eval_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {metric}: list(len={len(value)})")
            else:
                print(f"  {metric}: {value}")

def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
           return torch.device("cpu")
    else:
        return torch.device(device_str)

def main():
    args = parse_args()
    config = load_config(args.config)

    device = get_device(str(config.get("training", {}).get("device", "auto")).lower())
    print(f"Using device: {device}")
    
    data_config = config.get("data", {})
    intervention_config = config.get("intervention", {})
    max_steps = config["training"].get("max_steps", None)
    
    print("Loading model and tokenizer...")
    base_model, tokenizer = load_model_and_tokenizer(config["model"]["name_or_path"], device)
    
    intervention_module_path = intervention_config.get("module")
    if intervention_module_path:
        import_module(intervention_module_path)

    intervention_name = intervention_config.get("name")
    intervention_params = intervention_config.get("params", {})
    print(f"Preparing intervention module: {intervention_name}...")
    model = get_intervention(
        intervention_name,
        base_model=base_model,
        tokenizer=tokenizer,
        **intervention_params
    )

    metadata = config.get("metadata", {})

    output_dir = f"outputs/{metadata.get('run_name', intervention_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("Loading dataset...")
    dataset_module_path = data_config.get("module")
    dataset_module = import_module(dataset_module_path)
    get_dataloader = getattr(dataset_module, "get_dataloader")
    train_dataloader = get_dataloader(config, tokenizer, split="train")
    eval_dataloader = get_dataloader(config, tokenizer, split="test")
    
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        eps=float(config["training"].get("adam_eps", 1e-4)),
    )

    print("Running initial evaluation...")
    initial_eval_results = evaluate(model, eval_dataloader, device, tokenizer=tokenizer)
    print("Initial Evaluation Results:")
    for metric, value in initial_eval_results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {metric}: list(len={len(value)})")
        else:
            print(f"  {metric}: {value}")
    
    print("Starting training...")
    epoch_losses = []
    epoch_metrics = []
    
    for epoch in tqdm(range(config["training"]["epochs"]), desc="Training Epochs"):
        loss, losses, eval_results = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            device,
            eval_fn=evaluate,
            eval_dataloader=eval_dataloader,
            eval_kwargs={"tokenizer": tokenizer},
            max_steps=max_steps
        )
        epoch_losses.append(loss)
        epoch_metrics.append(eval_results)
        # save eval examples for manual inspection
        if eval_results and "examples" in eval_results:
            save_eval_examples(eval_results["examples"], output_dir=output_dir, filename=f"eval_examples_epoch_{epoch+1}.csv")
        
        print_epoch_summary(epoch + 1, loss, eval_results)

    # Save checkpoint with trained steering embedding
    import os
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "steering_embedding": model.steering_embedding.data,
        "config": config,
    }
    torch.save(checkpoint, f"{output_dir}/checkpoint.pt")
    print(f"\nCheckpoint saved to {output_dir}/checkpoint.pt")

    # final plots and save
    plot_training_curves(epoch_losses, epoch_metrics, output_dir=output_dir)

if __name__ == "__main__":
    main()
