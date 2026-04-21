"""Entry point for the vertical-slice evaluation (Step 3 of thesis_plan.md §9).

Runs 2 models × 3 FAE methods × 3 metrics on the test split and prints
a summary of the resulting DataFrame.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

# Ensure the project root is on sys.path so `src.*` imports resolve.
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import run_vertical_slice  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertical-slice evaluation")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable attribution disk cache")
    args = parser.parse_args()

    t0 = time.time()

    df = run_vertical_slice(
        data_root="data",
        resnet_weights="resnet_skin.pth",
        squeezenet_weights="squeezenet_skin.pth",
        split="test",
        device=None,
        output_csv="results/vertical_slice.csv",
        seed=42,
        use_cache=not args.no_cache,
    )

    elapsed = time.time() - t0

    # --- Summary ---
    print("\n" + "=" * 72)
    print("VERTICAL SLICE SUMMARY")
    print("=" * 72)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total runtime:   {elapsed:.1f} s\n")

    # Pivot to show per-(model, fae_method, metric) statistics
    summary = (
        df.groupby(["model", "fae_method", "metric"])["score"]
        .agg(["mean", "std", lambda x: x.isna().sum()])
        .rename(columns={"<lambda_0>": "nan_count"})
    )
    print("Per-(model, fae_method, metric) summary:")
    print(summary.to_string())

    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print(f"\nCSV written to: results/vertical_slice.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
