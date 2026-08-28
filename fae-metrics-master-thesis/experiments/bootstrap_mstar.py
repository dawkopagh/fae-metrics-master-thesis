"""bootstrap_mstar.py — image-bootstrap stability of the pruned metric set M*.

Answers whether the reported M* (and the cross-model identity that is the
thesis's headline redundancy finding) is an artifact of the particular
600-image draw: resamples the test images with replacement (per model),
re-runs the variance pre-screen + Spearman redundancy pruning, and counts
how often the point-estimate M* and the cross-model identity are reproduced.

Result with seed 42, B=200 (2026-08-28): M* reproduced 200/200 on both
models; cross-model identity holds 200/200. Cited in Chapter 4 next to the
tie-break disclosure.

Usage:
    .venv/bin/python experiments/bootstrap_mstar.py [--replicates 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.metrics.redundancy import (  # noqa: E402
    compute_redundancy_matrix,
    prune_redundant_metrics,
)

_RESULTS = _REPO.parent / "results"


def mstar_for(df_model: pd.DataFrame, model: str) -> frozenset:
    mat = compute_redundancy_matrix(df_model)[(model,)]
    kept, _ = prune_redundant_metrics(mat)
    return frozenset(kept)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--replicates", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slice-csv",
                   default=str(_RESULTS / "full_run_7fae_12metrics_600.csv"))
    args = p.parse_args()

    raw = pd.read_csv(args.slice_csv)
    rng = np.random.default_rng(args.seed)
    models = ["resnet18", "squeezenet"]
    imgs = {m: raw[raw.model == m].image_id.unique() for m in models}

    base = {m: mstar_for(raw[raw.model == m], m) for m in models}
    print("point-estimate M*:", {m: sorted(v) for m, v in base.items()})
    print("cross-model identity:", base[models[0]] == base[models[1]])

    same = {m: 0 for m in models}
    identity = 0
    for _ in range(args.replicates):
        boots = {}
        for m in models:
            ids = rng.choice(imgs[m], size=len(imgs[m]), replace=True)
            # Each draw needs its own image_id: the redundancy pivot keys on
            # (image_id, fae_method) with aggfunc="first", so re-drawn images
            # sharing an id would be collapsed to a single row and the
            # resample would silently degenerate into a ~63%-unique
            # subsample instead of a bootstrap.
            draw = pd.DataFrame({"image_id": ids,
                                 "_draw": np.arange(len(ids))})
            boot = draw.merge(raw[raw.model == m], on="image_id", how="left")
            boot["image_id"] = (boot["image_id"].astype(str) + "#"
                                + boot["_draw"].astype(str))
            boot = boot.drop(columns="_draw")
            boots[m] = mstar_for(boot, m)
            if boots[m] == base[m]:
                same[m] += 1
        if boots[models[0]] == boots[models[1]]:
            identity += 1

    B = args.replicates
    print(f"B={B}: M* == point estimate: "
          f"resnet {same['resnet18']}/{B}, squeeze {same['squeezenet']}/{B}; "
          f"cross-model identity {identity}/{B}")


if __name__ == "__main__":
    main()
