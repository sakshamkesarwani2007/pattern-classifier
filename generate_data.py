"""Make synthetic wallpaper-pattern images.
Idea: draw a random motif on one tile, copy it through every operation of the group,
then show a zoomed / rotated view of the repeating plane.
"""
import argparse
import os

import numpy as np
from PIL import Image

from groups import GROUPS, GROUP_NAMES, group_ops, sample_lattice

SPLITS = {"train": 0, "val": 1, "test": 2}


def make_motif(rng):
    n = rng.integers(4, 8)
    return {
        "bg": rng.uniform(0, 1, 3),
        "centres": rng.uniform(0, 1, (n, 2)),
        "widths": rng.uniform(0.05, 0.18, n),
        "colours": rng.uniform(0, 1, (n, 3)),
    }


def motif_colour(motif, u):
    out = np.tile(motif["bg"], (len(u), 1))
    for c, w, col in zip(motif["centres"], motif["widths"], motif["colours"]):
        d = (u - c + 0.5) % 1.0 - 0.5
        weight = np.exp(-(d ** 2).sum(1) / (2 * w ** 2))
        out += weight[:, None] * (col - motif["bg"])
    return out


def render(name, size, rng):
    lattice_type = GROUPS[name][0]
    basis = sample_lattice(lattice_type, rng)   
    motif = make_motif(rng)
    ops = group_ops(name)

    cells = rng.uniform(2.5, 5.0)              
    angle = np.radians(rng.uniform(-10, 10))
    shift = rng.uniform(0, 1, 2) @ basis
    ys, xs = np.mgrid[0:size, 0:size]
    p = np.stack([xs.ravel(), ys.ravel()], 1) / size * cells - cells / 2
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    p = p @ rot.T + shift
    u = p @ np.linalg.inv(basis)                

    img = np.zeros((len(u), 3))
    for M, t in ops:
        img += motif_colour(motif, u @ M.T + t)
    img /= len(ops)

    kind = rng.random()
    if kind < 0.25 or kind > 0.85:
        grey = img.mean(1, keepdims=True)
        if kind > 0.85:
            grey = (grey > np.median(grey)).astype(float)
        img = np.repeat(grey, 3, axis=1)

    # averaging pulls colours towards grey, so stretch back to the full range
    lo, hi = np.percentile(img, [2, 98])
    img = (img - lo) / (hi - lo + 1e-6)

    img = (img - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.1, 0.1)
    img += rng.normal(0, rng.uniform(0, 0.05), img.shape)
    img = np.clip(img, 0, 1).reshape(size, size, 3)
    return (img * 255).astype(np.uint8)

    
def make_split(out_dir, split, per_class, size, seed):
    for idx, name in enumerate(GROUP_NAMES):
        folder = os.path.join(out_dir, split, name)
        os.makedirs(folder, exist_ok=True)
        for i in range(per_class):
            rng = np.random.default_rng([seed, SPLITS[split], idx, i])
            img = render(name, size, rng)
            Image.fromarray(img).save(os.path.join(folder, f"{i:04d}.png"))
    print(split, "done:", per_class * len(GROUP_NAMES), "images")

def make_preview(path, size, seed):
    cols = 6
    rows = -(-len(GROUP_NAMES) // cols)
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for idx, name in enumerate(GROUP_NAMES):
        rng = np.random.default_rng([seed, 99, idx])
        tile = Image.fromarray(render(name, size, rng))
        sheet.paste(tile, ((idx % cols) * size, (idx // cols) * size))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path)
    print("saved", path)
    print("order:", ", ".join(GROUP_NAMES))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if args.preview:
        make_preview("results/preview.png", args.size, args.seed)
    else:
        make_split(args.out, "train", args.train, args.size, args.seed)
        make_split(args.out, "val", args.val, args.size, args.seed)
        make_split(args.out, "test", args.test, args.size, args.seed)
