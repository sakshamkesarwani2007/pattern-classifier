"""Classify the wallpaper group of an image.

Usage:
    python predict.py photo.jpg                  # asks both models
    python predict.py photo.jpg --model resnet
    python predict.py a.png b.png --top 5

"""
import argparse
import os

import torch

from dataset import load_image
from groups import IDX_TO_NAME
from models import get_model


def load_trained(name, device):
    path = f"checkpoints/{name}.pt"
    if not os.path.exists(path):
        return None
    model = get_model(name, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    model.eval()
    return model


def predict(model, image_path, device, top=3):
    x = load_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    values, indices = probs.topk(top)
    return [(IDX_TO_NAME[i], p) for i, p in zip(indices.tolist(), values.tolist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", choices=["resnet", "cnn", "both"], default="both")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ["cnn", "resnet"] if args.model == "both" else [args.model]
    models = {}
    for name in names:
        model = load_trained(name, device)
        if model is None:
            print(f"no checkpoint for {name}, run: python train.py --model {name}")
        else:
            models[name] = model

    for path in args.images:
        print(f"\n{path}")
        for name, model in models.items():
            results = predict(model, path, device, args.top)
            best, conf = results[0]
            print(f"  {name:6s} -> {best}  (confidence {conf:.1%})")
            if args.top > 1:
                print("         also: " + ", ".join(f"{g} {p:.1%}" for g, p in results[1:]))

if __name__ == "__main__":
    main()
