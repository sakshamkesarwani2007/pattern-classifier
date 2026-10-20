"""Test a trained model on the held-out synthetic test set.

Usage:
    python evaluate.py --model cnn
    python evaluate.py --model resnet

"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from dataset import get_loaders
from groups import GROUP_NAMES
from models import get_model

def predict_all(model, loader, device):
    preds, truth = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images.to(device))
            preds.extend(scores.argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    return np.array(truth), np.array(preds)

def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(GROUP_NAMES)))
    ax.set_yticks(range(len(GROUP_NAMES)))
    ax.set_xticklabels(GROUP_NAMES, rotation=90)
    ax.set_yticklabels(GROUP_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] > 0:
                colour = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=colour, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet", "cnn"], default="resnet")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(f"checkpoints/{args.model}.pt", map_location=device)
    model = get_model(args.model, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, _, test_loader = get_loaders(batch_size=args.batch_size)
    truth, preds = predict_all(model, test_loader, device)

    accuracy = (truth == preds).mean()
    cm = confusion_matrix(truth, preds, labels=range(len(GROUP_NAMES)))
    per_class = {name: float(cm[i, i] / max(cm[i].sum(), 1)) for i, name in enumerate(GROUP_NAMES)}

    print(f"{args.model} test accuracy: {accuracy:.3f} on {len(truth)} images")
    for name, acc in per_class.items():
        print(f"  {name:5s} {acc:.2f}")

    off_diag = [(cm[i, j], GROUP_NAMES[i], GROUP_NAMES[j])
                for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    print("most common mistakes (count, true -> predicted):")
    for count, true, pred in sorted(off_diag, reverse=True)[:5]:
        print(f"  {count}  {true} -> {pred}")

    plot_confusion(cm, f"results/{args.model}_confusion.png",
                   f"{args.model}: test accuracy {accuracy:.3f}")
    with open(f"results/{args.model}_metrics.json", "w") as f:
        json.dump({"test_accuracy": float(accuracy), "per_class": per_class}, f, indent=2)
    print(f"saved results/{args.model}_confusion.png and results/{args.model}_metrics.json")


if __name__ == "__main__":
    main()
