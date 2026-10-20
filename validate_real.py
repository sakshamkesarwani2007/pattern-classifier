"""Score the trained models on the hand-labelled real patterns in data/real.

Usage:
    python validate_real.py
    python validate_real.py --model resnet --top 3
"""
import argparse
import csv
import os

import torch

from groups import NAME_TO_IDX
from predict import load_trained, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet", "cnn", "both"], default="both")
    ap.add_argument("--dir", default="data/real")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "labels.csv"), newline="") as f:
        rows = [(r["filename"], r["group"]) for r in csv.DictReader(f)]
    for _, group in rows:
        if group not in NAME_TO_IDX:
            raise SystemExit(f"unknown group in labels.csv: {group}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ["cnn", "resnet"] if args.model == "both" else [args.model]

    for name in names:
        model = load_trained(name, device)
        if model is None:
            print(f"no checkpoint for {name}, run: python train.py --model {name}")
            continue

        print(f"\n== {name} ==")
        top1 = topk = 0
        for filename, truth in rows:
            results = predict(model, os.path.join(args.dir, filename), device, args.top)
            guesses = [g for g, _ in results]
            ok1, okk = guesses[0] == truth, truth in guesses
            top1 += ok1
            topk += okk
            best, conf = results[0]
            print(f"  {filename:12s} true {truth:5s} pred {best:5s} ({conf:5.1%})  "
                  f"{'OK' if ok1 else ('top-%d' % args.top if okk else 'wrong')}")

        n = len(rows)
        print(f"  top-1 accuracy: {top1}/{n} = {top1 / n:.1%}")
        print(f"  top-{args.top} accuracy: {topk}/{n} = {topk / n:.1%}")


if __name__ == "__main__":
    main()
