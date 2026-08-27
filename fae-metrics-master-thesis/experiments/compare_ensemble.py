"""compare_ensemble.py — individual vs. NormEnsembleXAI-ensembled effectiveness.

Generates ``results/individual_vs_ensemble.csv`` (and the per-source variant).

Normalization modes
-------------------
``pooled`` (default, the PRIMARY comparison reported in the thesis):
    One Second-Moment-Scaling RMS per (model, metric), computed over the
    individual-method scores and the ensemble scores POOLED, so both
    populations share a common scale. This is the comparison that can see a
    level difference between the ensemble and the individuals.
``per-source`` (robustness variant; the pre-2026-08 methodology):
    Individual rows are scaled by the RMS of the individual population,
    ensemble rows by the RMS of the ensemble's own scores. Because
    mean(x)/RMS(x) = 1/sqrt(1+CV^2) for a positive-valued metric, this
    variant compares dispersion profiles and is blind to uniform level
    differences — reported only to document normalization-conditionality.
``winsorized-pooled`` (robustness variant):
    As ``pooled``, but each (model, metric) pool is clipped at its 1st/99th
    percentile before the RMS is computed and applied. Robust to the
    heavy-tailed raw scores documented in Chapter 5 (RMA mass fractions > 1,
    LRP max-sensitivity explosions), which otherwise inflate the pooled RMS.
    Source of the winsorized numbers cited in Section 4 (exp-ensemble).

Baselines (all per (model, image), uniform weights over the seven M* metrics,
NaN-aware):
    eff_individual_mean       — mean over the seven individual methods.
    eff_individual_best       — per-image ORACLE: max over the seven methods
                                (switches methods image-by-image; includes
                                Occlusion, which is not an ensemble member).
    eff_best_fixed            — the single fixed method with the highest mean
                                effectiveness on that model (name recorded in
                                ``best_fixed_method``).
    eff_best_fixed_gradient   — the best fixed GRADIENT method (an actual
                                ensemble member; name in
                                ``best_fixed_gradient_method``).

Usage:
    .venv/bin/python experiments/compare_ensemble.py                 # pooled
    .venv/bin/python experiments/compare_ensemble.py --normalization per-source
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

GRADIENT_METHODS = [
    "integrated_gradients", "saliency", "gradcam",
    "guided_backprop", "deep_lift", "lrp",
]


def _sms(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """NaN-aware Second Moment Scaling + direction rectification.

    ``mode='pooled'``: RMS per (model, metric) over ALL rows (individual +
    ensemble). ``mode='per-source'``: RMS per (model, metric, source) where
    source distinguishes ensemble rows from individual rows.
    ``mode='winsorized-pooled'``: as pooled, but the pool is clipped at its
    1st/99th percentile before the RMS is computed and applied (the clipped
    values are what get normalized).
    """
    out = df.copy()
    out["_source"] = np.where(out.fae_method == "ensemble", "ens", "ind")
    keys = (["model", "metric", "_source"] if mode == "per-source"
            else ["model", "metric"])
    out["norm"] = np.nan
    for _, idx in out.groupby(keys).groups.items():
        raw = out.loc[idx, "score"].values.astype(float)
        if mode == "winsorized-pooled":
            finite = raw[~np.isnan(raw)]
            if len(finite):
                lo, hi = np.percentile(finite, [1, 99])
                raw = np.clip(raw, lo, hi)
        valid = raw[~np.isnan(raw)]
        rms = np.sqrt(np.mean(valid**2)) if len(valid) else np.nan
        metric = out.loc[idx, "metric"].iloc[0]
        if not rms or np.isnan(rms):
            out.loc[idx, "norm"] = 0.0
        else:
            out.loc[idx, "norm"] = (raw / rms) * METRIC_DIRECTIONS[metric]
    return out.drop(columns="_source")


def build(mode: str) -> pd.DataFrame:
    ind = pd.read_csv(_RESULTS / "full_run_7fae_12metrics_600.csv")
    ens = pd.read_csv(_RESULTS / "full_run_ensemble_7fae_12metrics_600.csv")
    df = pd.concat([ind, ens], ignore_index=True)
    df = _sms(df[df.metric.isin(M_STAR)], mode=mode)

    eff = (df.groupby(["model", "image_id", "fae_method"])["norm"]
           .mean().reset_index())

    rows = []
    for model, sub in eff.groupby("model"):
        e = sub[sub.fae_method == "ensemble"].set_index("image_id")["norm"]
        i = sub[sub.fae_method != "ensemble"]
        g = i.groupby("image_id")["norm"]
        per_method = i.groupby("fae_method")["norm"].mean()
        best_fixed = per_method.idxmax()
        best_grad = per_method[GRADIENT_METHODS].idxmax()
        fx = i[i.fae_method == best_fixed].set_index("image_id")["norm"]
        fg = i[i.fae_method == best_grad].set_index("image_id")["norm"]
        out = pd.DataFrame({
            "eff_ensemble": e,
            "eff_individual_mean": g.mean(),
            "eff_individual_best": g.max(),
            "eff_best_fixed": fx,
            "eff_best_fixed_gradient": fg,
        }).reset_index()
        out.insert(0, "model", model)
        out["best_fixed_method"] = best_fixed
        out["best_fixed_gradient_method"] = best_grad
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


_DEFAULT_NAMES = {
    "pooled": "individual_vs_ensemble.csv",
    "per-source": "individual_vs_ensemble_per_source.csv",
    "winsorized-pooled": "individual_vs_ensemble_winsorized.csv",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--normalization", choices=sorted(_DEFAULT_NAMES),
                   default="pooled")
    p.add_argument("--output-csv", default=None,
                   help="Default: results/" + ", ".join(
                       f"{v} ({k})" for k, v in _DEFAULT_NAMES.items()))
    args = p.parse_args()

    out_path = (Path(args.output_csv) if args.output_csv
                else _RESULTS / _DEFAULT_NAMES[args.normalization])

    out = build(args.normalization)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows, normalization={args.normalization})")
    from scipy.stats import wilcoxon
    for model, sub in out.groupby("model"):
        w, p_val = wilcoxon(sub.eff_ensemble, sub.eff_individual_mean)
        print(f"  {model}: ens={sub.eff_ensemble.mean():.3f} "
              f"mean={sub.eff_individual_mean.mean():.3f} "
              f"(ens-vs-mean W={w:.0f} p={p_val:.2e}) "
              f"oracle={sub.eff_individual_best.mean():.3f} "
              f"best_fixed={sub.eff_best_fixed.mean():.3f} "
              f"({sub.best_fixed_method.iloc[0]}) "
              f"best_grad={sub.eff_best_fixed_gradient.mean():.3f} "
              f"({sub.best_fixed_gradient_method.iloc[0]})")


if __name__ == "__main__":
    main()
