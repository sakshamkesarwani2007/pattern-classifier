# Wallpaper Pattern Classifier

Classifies a repeating 2D pattern into one of the **17 wallpaper groups** (p1, p2, pm, pg, cm, pmm, pmg, pgg, cmm, p4, p4mm, p4gm, p3, p3m1, p31m, p6, p6mm).

There is no labelled dataset of real patterns, so the models are trained on **synthetic images**: a random motif is copied through every symmetry operation of a group and rendered as a zoomed, slightly rotated view of the repeating plane. A small set of hand-labelled real patterns is used to check how well this transfers.

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python generate_data.py                   # make data/synthetic (train/val/test)
python generate_data.py --preview         # one sample per group -> results/preview.png
python train.py --model resnet            # or --model cnn; saves checkpoints/<model>.pt
python evaluate.py --model resnet         # synthetic test accuracy + confusion matrix
python predict.py "my pattern.png"        # classify your own image (both models)
python predict.py a.png b.png --model resnet --top 5
python validate_real.py                   # score the models on data/real
```

Quote file names that contain spaces.

## Files

| File | Purpose |
|---|---|
| [groups.py](groups.py) | The 17 groups as generators (rotations, mirrors, glides) and the lattice each one needs. `python groups.py` checks every group has the right number of operations. |
| [generate_data.py](generate_data.py) | Renders synthetic patterns and the preview sheet. |
| [dataset.py](dataset.py) | Image loading and the train/val/test data loaders. |
| [models.py](models.py) | Pretrained ResNet-18 and a small 4-block CNN trained from scratch. |
| [train.py](train.py) | Training loop; keeps the checkpoint with the best validation accuracy. |
| [evaluate.py](evaluate.py) | Test-set accuracy, per-class accuracy, most common mistakes, confusion matrix. |
| [predict.py](predict.py) | Classify one or more images and show the top guesses. |
| [validate_real.py](validate_real.py) | Compare predictions with the labels in `data/real/labels.csv`. |

Generated folders: `data/synthetic/`, `checkpoints/`, `results/`.

## Real patterns

`data/real/` holds 17 real pattern images (`real01.png` to `real17.png`) and `labels.csv` (`filename,group`). The labels were assigned by hand and are not guaranteed to be correct, and several of these groups are hard to tell apart by eye.

### Verifying real01 to real17

Both models must be trained first (`python train.py --model cnn` and `python train.py --model resnet`). Run these from the project folder.

Score all 17 images against `labels.csv` (prints each prediction, then top-1 and top-3 accuracy):

```
python validate_real.py
```

Options:

```
python validate_real.py --model resnet     # one model only (cnn, resnet or both)
python validate_real.py --top 5            # widen the "top-k" check
```

Classify a single image or all 17 without comparing to the labels:

```
python predict.py data/real/real01.png --model resnet
python predict.py data/real/real*.png --model resnet --top 3
```

In PowerShell, the wildcard may not expand, so list the files with:

```
python predict.py (Get-ChildItem data/real/real*.png).FullName --model resnet
```

## Results

Synthetic test set (510 images):

| Model | Accuracy |
|---|---|
| Small CNN | 52.7% |
| ResNet-18 (pretrained) | 73.3% |

Real patterns (`validate_real.py`, 17 images):

| Model | Top-1 | Top-3 |
|---|---|---|
| Small CNN | 1/17 (5.9%) | 4/17 (23.5%) |
| ResNet-18 | 3/17 (17.6%) | 7/17 (41.2%) |

Chance is 5.9% (1 in 17). Both models are far weaker on real images than on synthetic ones. The real patterns are small, blurry, flat-colour motifs with a different look from the random blobs used for training, so the models do not transfer well. Ideas for improving this: blur, palette and resolution augmentation, more varied motifs in `generate_data.py`, and more real labelled images.
