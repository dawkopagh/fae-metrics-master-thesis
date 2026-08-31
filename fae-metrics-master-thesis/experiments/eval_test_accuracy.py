#!/usr/bin/env python
"""eval_test_accuracy.py — Score the trained classifiers on the test split.

LOCAL post-processing script (venv, NOT Colab). The training loop of
src/models/train.py iterates over the train and validation phases only: it
builds a test DataLoader but never scores it, so no classification test
accuracy existed for Table 4.1. This script supplies it, reusing the
evaluation-time pipeline exactly — ImageFolder over data/images/<split> with
the ._* resource-fork filter, Resize(224, 224) + ToTensor + ImageNet
normalisation, no augmentation — so the numbers are comparable with the
validation accuracies in the training logs.

Backbones are built with weights=None because the fine-tuned state_dict is
loaded over them; no ImageNet download is required. Running with
--split validation reproduces the logged best-validation accuracies
(0.7533 / 0.7400), which is the check that this pipeline matches training.

Run from the repo root:
    fae-metrics-master-thesis/.venv/bin/python \
        fae-metrics-master-thesis/experiments/eval_test_accuracy.py

Writes results/test_set_accuracy.csv (one row per model): accuracy, balanced
accuracy, macro F1, per-class recall/precision/F1, and the confusion matrix.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
WEIGHTS = [("resnet18", "resnet18_isic2017.pth"),
           ("squeezenet", "squeezenet_isic2017.pth")]


def _loader(split: str, batch_size: int, workers: int):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(
        os.path.join(REPO, "data/images", split), tf,
        is_valid_file=lambda p: not os.path.basename(p).startswith("._"),
    )
    train_classes = datasets.ImageFolder(
        os.path.join(REPO, "data/images/train"), tf,
        is_valid_file=lambda p: not os.path.basename(p).startswith("._"),
    ).classes
    # Label indices are positional: the split must enumerate classes in the
    # same order the model was trained on, or every score would be garbage.
    assert ds.classes == train_classes, (ds.classes, train_classes)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=workers)


def _build(arch: str) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif arch == "squeezenet":
        m = models.squeezenet1_1(weights=None)
        m.classifier[1] = nn.Conv2d(512, 3, kernel_size=1, stride=1)
        m.num_classes = 3
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["train", "validation", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="CSV path (default: results/<split>_set_accuracy.csv)")
    args = ap.parse_args()

    ds, loader = _loader(args.split, args.batch_size, args.num_workers)
    classes = ds.classes
    print(f"{args.split}: {len(ds)} images, classes {classes}")

    rows = []
    for arch, wfile in WEIGHTS:
        model = _build(arch)
        sd = torch.load(os.path.join(REPO, "weights", wfile), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model.eval()

        conf = torch.zeros(3, 3, dtype=torch.long)  # rows = true, cols = pred
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(1)
                for t, p in zip(y.tolist(), pred.tolist()):
                    conf[t, p] += 1

        n = conf.sum().item()
        correct = conf.diag().sum().item()
        acc = correct / n
        recalls, precisions, f1s = [], [], []
        for c in range(3):
            tp = conf[c, c].item()
            rec = tp / conf[c].sum().item() if conf[c].sum() else 0.0
            pre = tp / conf[:, c].sum().item() if conf[:, c].sum() else 0.0
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
            recalls.append(rec); precisions.append(pre); f1s.append(f1)

        print(f"\n=== {arch} ({args.split}) ===")
        print(f"accuracy {acc:.4f} ({correct}/{n})  "
              f"balanced {np.mean(recalls):.4f}  macro-F1 {np.mean(f1s):.4f}")
        for c, name in enumerate(classes):
            print(f"  {name:22s} n={conf[c].sum().item():4d}  "
                  f"recall {recalls[c]:.3f}  precision {precisions[c]:.3f}")

        rows.append({
            "model": arch, "split": args.split, "n": n, "correct": correct,
            "accuracy": round(acc, 6),
            "balanced_accuracy": round(float(np.mean(recalls)), 6),
            "macro_f1": round(float(np.mean(f1s)), 6),
            **{f"recall_{classes[c]}": round(recalls[c], 6) for c in range(3)},
            **{f"precision_{classes[c]}": round(precisions[c], 6) for c in range(3)},
            **{f"f1_{classes[c]}": round(f1s[c], 6) for c in range(3)},
            "confusion_row_true": ";".join(",".join(map(str, r))
                                           for r in conf.tolist()),
        })

    out = args.out or os.path.join(
        os.path.dirname(REPO), "results", f"{args.split}_set_accuracy.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
