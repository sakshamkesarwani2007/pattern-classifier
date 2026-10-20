"""Train a model on the synthetic patterns.
Usage:
    python train.py --model cnn
    python train.py --model resnet
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from dataset import get_loaders
from models import get_model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            loss = loss_fn(scores, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            seen += len(labels)

    return total_loss / seen, correct / seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet", "cnn"], default="resnet")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = args.lr or (1e-4 if args.model == "resnet" else 1e-3)

    train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size)
    model = get_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    ckpt_path = f"checkpoints/{args.model}.pt"
    history = []
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch}/{args.epochs}  train loss {train_loss:.3f} acc {train_acc:.3f}"
              f"  val loss {val_loss:.3f} acc {val_acc:.3f}  ({time.time() - start:.0f}s)")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": args.model, "state_dict": model.state_dict()}, ckpt_path)

    with open(f"results/{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val acc {best_val:.3f}, saved to {ckpt_path}")


if __name__ == "__main__":
    main()
