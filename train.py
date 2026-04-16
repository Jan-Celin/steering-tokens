import argparse
from importlib import import_module
import torch
from utils.config import load_config
from models.model_manager import load_model_and_tokenizer
from interventions import get_intervention
from training.trainer import train_epoch
from training.metrics import evaluate_epoch
from utils.visualization import plot_training_curves, print_epoch_summary
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_tokens.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    requested_device = str(config.get("training", {}).get("device", "auto")).lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            print("Requested device 'cuda' is unavailable. Falling back to 'mps'.")
            device = "mps"
        else:
            print("Requested device 'cuda' is unavailable. Falling back to 'cpu'.")
            device = "cpu"
    elif requested_device == "mps":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            print("Requested device 'mps' is unavailable. Falling back to 'cuda'.")
            device = "cuda"
        else:
            print("Requested device 'mps' is unavailable. Falling back to 'cpu'.")
            device = "cpu"
    elif requested_device == "cpu":
        device = "cpu"
    else:
        print(f"Unknown device '{requested_device}'. Falling back to auto selection.")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")
    data_config = config.get("data", {})
    intervention_config = config.get("intervention", {})
    
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
    
    print("Loading dataset...")
    dataset_module_path = data_config.get("module")
    dataset_module = import_module(dataset_module_path)
    get_dataloader = getattr(dataset_module, "get_dataloader")
    train_dataloader = get_dataloader(config, tokenizer, split="train")
    
    # Also load eval dataloader (same data for now, but could use separate split)
    eval_dataloader = get_dataloader(config, tokenizer, split="train")
    
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        eps=float(config["training"].get("adam_eps", 1e-4)),
    )
    
    print("Starting training...")
    epoch_losses = []
    epoch_metrics = []
    
    for epoch in tqdm(range(config["training"]["epochs"]), desc="Training Epochs"):
        loss, losses, eval_results = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            device,
            eval_fn=evaluate_epoch,
            eval_dataloader=eval_dataloader,
            eval_kwargs={"tokenizer": tokenizer},
        )
        epoch_losses.append(loss)
        epoch_metrics.append(eval_results)
        
        print_epoch_summary(epoch + 1, loss, eval_results)
    
    # Generate plots
    print("\nGenerating training visualizations...")
    plot_training_curves(epoch_losses, epoch_metrics, output_dir="outputs")
    
    print("\nTraining complete!")
    print(f"Final loss: {epoch_losses[-1]:.4f}")
    print(f"Final mean MSE: {epoch_metrics[-1]['mean_mse']:.6f}")
    print(f"Final exact match rate: {epoch_metrics[-1]['exact_match_rate']:.2%}")

if __name__ == "__main__":
    main()
    plt.ylabel("Loss")
    plt.title("Training Loss")
    # save the plot and name it with the current timestamp
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    plt.savefig(f"outputs/training_loss_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()

if __name__ == "__main__":
    main()
