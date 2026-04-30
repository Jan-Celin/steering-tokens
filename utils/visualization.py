import matplotlib.pyplot as plt
import csv
import os

def plot_training_curves(epoch_losses, epoch_metrics, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(epoch_losses) + 1))

    # Loss plot
    plt.figure()
    plt.plot(epochs, epoch_losses, marker="o", label="train_loss")
    eval_losses = [m["mean_loss"] if m is not None else None for m in epoch_metrics]
    if any(x is not None for x in eval_losses):
        plt.plot(epochs, eval_losses, marker="o", label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per epoch")
    plt.savefig(os.path.join(output_dir, "loss_per_epoch.png"))
    plt.close()

    # Accuracy / exact match
    plt.figure()
    accs = [m["token_accuracy"] if m is not None else None for m in epoch_metrics]
    em = [m["exact_match_rate"] if m is not None else None for m in epoch_metrics]
    if any(x is not None for x in accs):
        plt.plot(epochs, accs, marker="o", label="token_accuracy")
    if any(x is not None for x in em):
        plt.plot(epochs, em, marker="o", label="exact_match_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Evaluation metrics per epoch")
    plt.savefig(os.path.join(output_dir, "metrics_per_epoch.png"))
    plt.close()


def save_eval_examples(examples, output_dir="outputs", filename="eval_examples.csv"):
    """examples: iterable of dicts with keys: source_text, target_text, predicted_text"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_text", "target_text", "predicted_text"])
        writer.writeheader()
        for ex in examples:
            writer.writerow({
                "source_text": ex.get("source_text", ""),
                "target_text": ex.get("target_text", ""),
                "predicted_text": ex.get("predicted_text", ""),
            })
