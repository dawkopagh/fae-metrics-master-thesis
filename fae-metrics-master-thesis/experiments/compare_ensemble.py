"""compare_ensemble.py — individual vs. NormEnsembleXAI-ensembled effectiveness.

Reconstructs (and now maintains) the generator of
``results/individual_vs_ensemble.csv``, which was previously produced by
ad-hoc code that never landed in the repo. The methodology is calibrated to
reproduce the 2026-06-25 CSV exactly (max |diff| = 0.0 per column) when run
with the pre-fix metric directions:

- Per (model, metric), scores are Second-Moment-Scaled with the RMS of the
  SOURCE's own score distribution: individual-method rows use the RMS over
  the individual full-run scores; ensemble rows use the RMS over the
  ensemble run's scores. Each effectiveness index is therefore
  scale-normalized within its own score population.
- Direction rectification via METRIC_DIRECTIONS (src/aggregation/normalize).
- Effectiveness per (model, image, method) = nan-aware uniform mean over the
  seven M* metrics.
- ``eff_individual_mean`` / ``eff_individual_best`` = mean / max over the
  seven individual FAE methods, paired per image.

Usage (defaults point at the authoritative full-run CSVs):

    .venv/bin/python experiments/compare_ensemble.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.aggregation.normalize import METRIC_DIRECTIONS  # noqa: E402

_RESULTS = _REPO.parent / "results"

M_STAR = [
    "faithfulness_correlation",
    "max_sensitivity",
    "pixel_flipping",
    "pointing_game",
    "random_logit",
    "relevance_mass_accuracy",
    "sparseness",
]


def _sms_within_source(df: pd.DataFrame) -> pd.DataFrame:
    """NaN-aware Second Moment Scaling per (model, metric) within *df*."""
    out = df.copy()
    out["norm"] = np.nan
    for (_, metric), idx in out.groupby(["model", "metric"]).groups.items():
        raw = out.loc[idx, "score"].values.astype(float)
        valid = raw[~np.isnan(raw)]
        rms = np.sqrt(np.mean(valid**2)) if len(valid) else np.nan
        if not rms or np.isnan(rms):
            out.loc[idx, "norm"] = 0.0
        else:
            out.loc[idx, "norm"] = (raw / rms) * METRIC_DIRECTIONS[metric]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--individual-csv",
                   default=str(_RESULTS / "full_run_7fae_12metrics_600.csv"))
    p.add_argument("--ensemble-csv",
                   default=str(_RESULTS / "full_run_ensemble_7fae_12metrics_600.csv"))
    p.add_argument("--output-csv",
                   default=str(_RESULTS / "individual_vs_ensemble.csv"))
    args = p.parse_args()

    ind = pd.read_csv(args.individual_csv)
    ens = pd.read_csv(args.ensemble_csv)
    ind = _sms_within_source(ind[ind.metric.isin(M_STAR)])
    ens = _sms_within_source(ens[ens.metric.isin(M_STAR)])

    eff_ind = (ind.groupby(["model", "image_id", "fae_method"])["norm"]
               .mean().reset_index())
    g = eff_ind.groupby(["model", "image_id"])["norm"]
    eff_ens = (ens.groupby(["model", "image_id"])["norm"]
               .mean().rename("eff_ensemble"))

    out = pd.DataFrame({
        "eff_ensemble": eff_ens,
        "eff_individual_mean": g.mean(),
        "eff_individual_best": g.max(),
    }).reset_index()
    out.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv} ({len(out)} rows)")
    for model, sub in out.groupby("model"):
        print(f"  {model}: ens={sub.eff_ensemble.mean():.3f} "
              f"mean_ind={sub.eff_individual_mean.mean():.3f} "
              f"best_ind={sub.eff_individual_best.mean():.3f}")


if __name__ == "__main__":
    main()
