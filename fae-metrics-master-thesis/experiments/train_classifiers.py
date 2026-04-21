"""Train ResNet-18 and SqueezeNet 1.1 on ISIC 2017.

Usage::

    python experiments/train_classifiers.py \\
        --data_root data/isic2017 \\
        --output_dir weights \\
        --device cuda \\
        --epochs 25 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.train import train_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ISIC 2017 classifiers")
    parser.add_argument("--data_root", type=str, default="data/isic2017",
                        help="Root dir with images/{train,validation,test}/{class}/")
    parser.add_argument("--output_dir", type=str, default="weights",
                        help="Directory for saved weight files")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if omitted)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    architectures = [
        ("resnet18", f"{args.output_dir}/resnet18_isic2017.pth"),
        ("squeezenet", f"{args.output_dir}/squeezenet_isic2017.pth"),
    ]

    results = {}
    t0 = time.time()

    for arch, weights_path in architectures:
        print(f"\n{'='*60}")
        print(f"Training {arch}")
        print(f"{'='*60}\n")

        log_df = train_classifier(
            arch=arch,
            data_root=args.data_root,
            output_weights_path=weights_path,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # Save per-epoch log
        log_path = Path("results") / f"training_log_{arch}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)

        best_val = log_df[log_df["phase"] == "validation"]["acc"].max()
        results[arch] = best_val

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f} s")
    for arch, acc in results.items():
        print(f"  {arch:15s}  best val acc = {acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
