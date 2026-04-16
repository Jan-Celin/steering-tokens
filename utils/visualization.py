import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


def plot_training_curves(
    epoch_losses: List[float],
    epoch_metrics: List[Dict[str, float]],
    output_dir: str = "outputs",
) -> None:
    """
    Create training visualization with loss curve and MSE metrics.
    
    Args:
        epoch_losses: List of epoch-level average losses
        epoch_metrics: List of dicts with eval metrics per epoch
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(epoch_losses) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight='bold')
    
    # Plot 1: Training loss
    ax = axes[0, 0]
    ax.plot(epochs, epoch_losses, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss per Epoch")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MSE metrics
    ax = axes[0, 1]
    mean_mses = [m["mean_mse"] for m in epoch_metrics]
    std_mses = [m["std_mse"] for m in epoch_metrics]
    ax.plot(epochs, mean_mses, 'r-o', linewidth=2, markersize=6, label="Mean MSE")
    ax.fill_between(epochs, 
                     np.array(mean_mses) - np.array(std_mses),
                     np.array(mean_mses) + np.array(std_mses),
                     alpha=0.2, color='red', label="±1 Std Dev")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Mean Squared Error (with Std Dev)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Exact match rate
    ax = axes[1, 0]
    exact_match_rates = [m["exact_match_rate"] for m in epoch_metrics]
    ax.plot(epochs, exact_match_rates, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Exact Match Accuracy")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Boxplots of MSE per epoch
    ax = axes[1, 1]
    mse_values_by_epoch = [m["mse_values"] for m in epoch_metrics]
    bp = ax.boxplot(mse_values_by_epoch, labels=epochs, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("MSE Distribution (Boxplot)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "training_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()


def print_epoch_summary(epoch_num: int, loss: float, metrics: Dict[str, float]) -> None:
    """Pretty-print epoch summary."""
    print(f"\nEpoch {epoch_num} Summary:")
    print(f"  Loss:              {loss:.4f}")
    print(f"  Mean MSE:          {metrics['mean_mse']:.6f}")
    print(f"  Std MSE:           {metrics['std_mse']:.6f}")
    print(f"  Median MSE:        {metrics['median_mse']:.6f}")
    print(f"  Exact Match Rate:  {metrics['exact_match_rate']:.2%}")
    print(f"  Parse Errors:      {metrics['parse_error_count']}/{metrics['total_samples']}")
