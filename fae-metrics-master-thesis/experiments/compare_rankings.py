"""
compare_rankings.py — Part D of S10: compare four FAE ranking schemes.

Metric set M* (7 metrics, cross-model intersection; see the M_STAR constant
below, which is authoritative):
  faithfulness_correlation, max_sensitivity, pixel_flipping, pointing_game,
  random_logit, relevance_mass_accuracy, sparseness

Explicitly excluded from aggregation (not in M*):
  avg_sensitivity  — pruned (redundant with max_sensitivity)
  complexity       — pruned (redundant with sparseness)
  completeness     — structural pre-screen (zero variance)
  non_sensitivity  — structural pre-screen (disabled, all-NaN)
  model_parameter_randomisation — structural pre-screen (disabled on ISIC)

Four effectiveness schemes:
  (a) uniform      — 1/7 per metric, second-moment normalisation
  (b) autoweighted — CV-inverse weights per model, same normalisation
  (c) mqdiscount   — autoweighted × MetaQuantus reliability discount
                     (NR-only fallback for AR-inapplicable metrics)
  (d) single_fc    — faithfulness_correlation only (single-metric baseline)

Output: results/ranking_comparison.csv
  Columns: model, fae_method, image_id,
           effectiveness_uniform, effectiveness_autoweighted,
           effectiveness_mqdiscount, effectiveness_single_fc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.aggregation.weighting import (
    autoweighted_weights,
    metaquantus_discounted_weights,
)
from src.aggregation.normalize import METRIC_DIRECTIONS

# ---------------------------------------------------------------------------
# Default paths (12-image preliminary run)
# ---------------------------------------------------------------------------
_RESULTS = _REPO.parent / "results"
_DEFAULT_SLICE   = str(_RESULTS / "vertical_slice_7fae_12metrics.csv")
_DEFAULT_RELCSV  = str(_RESULTS / "meta_evaluation_reliability.csv")
_DEFAULT_OUT     = str(_RESULTS / "ranking_comparison.csv")

# ---------------------------------------------------------------------------
# M* metric set (S8.5 cross-model intersection, 7 metrics)
# ---------------------------------------------------------------------------
# M* — the non-redundant metric set from the full-run redundancy analysis
# (|rho|>0.85 pruning, identical for both models). Excludes the all-NaN
# model_parameter_randomisation (MPRT disabled on ISIC) and complexity/
# avg_sensitivity (pruned as redundant with sparseness / max_sensitivity).
M_STAR: list[str] = [
    "faithfulness_correlation",
    "max_sensitivity",
    "pixel_flipping",
    "pointing_game",
    "random_logit",
    "relevance_mass_accuracy",
    "sparseness",
]

# Quantus categories for mqdiscount category-fallback (step 2)
_CATEGORY_MAP: dict[str, str] = {
    "faithfulness_correlation": "faithfulness",
    "pixel_flipping": "faithfulness",
    "pointing_game": "localisation",
    "relevance_mass_accuracy": "localisation",
    "max_sensitivity": "robustness",
    "sparseness": "complexity",
    "model_parameter_randomisation": "randomisation",
    "random_logit": "randomisation",
}


# ---------------------------------------------------------------------------
# Normalisation (nan-aware second-moment scaling)
# ---------------------------------------------------------------------------

def _second_moment_nan_aware(scores: np.ndarray, direction: int) -> np.ndarray:
    """Second-moment scaling that preserves NaN positions.

    Uses nanmean so that NaN entries in the group do not corrupt the RMS
    denominator (avoids all-NaN normalised output for partially-NaN metrics
    such as model_parameter_randomisation).
    """
    s = np.asarray(scores, dtype=np.float64)
    rms = np.sqrt(np.nanmean(s ** 2))
    if rms == 0.0 or np.isnan(rms):
        return np.zeros_like(s)
    return (s / rms) * direction


def normalize_m_star(df: pd.DataFrame) -> pd.DataFrame:
    """Apply nan-aware second-moment normalisation within (model, metric)."""
    out = df.copy()
    out["score_normalized"] = np.nan
    for (model, metric), idx in out.groupby(["model", "metric"]).groups.items():
        direction = METRIC_DIRECTIONS[metric]
        raw = out.loc[idx, "score"].values
        out.loc[idx, "score_normalized"] = _second_moment_nan_aware(raw, direction)
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _weighted_mean_nan_aware(scores: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean over non-NaN entries; renormalises weights over valid set."""
    valid = ~np.isnan(scores)
    if not valid.any():
        return float("nan")
    w_valid = weights[valid]
    w_sum = w_valid.sum()
    if w_sum <= 0.0:
        return float("nan")
    return float(np.dot(w_valid / w_sum, scores[valid]))


def aggregate_group(
    group_df: pd.DataFrame,
    weights: dict[str, float],
    metric_order: list[str],
) -> float:
    """Compute effectiveness index for one (model, image_id, fae_method) group."""
    score_map = dict(zip(group_df["metric"], group_df["score_normalized"]))
    scores = np.array([score_map.get(m, np.nan) for m in metric_order])
    w = np.array([weights[m] for m in metric_order])
    return _weighted_mean_nan_aware(scores, w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Four-scheme FAE ranking comparison (Part D, S10)."
    )
    p.add_argument(
        "--slice-csv",
        default=_DEFAULT_SLICE,
        metavar="PATH",
        help=(
            "Vertical slice CSV to read "
            f"(default: {_DEFAULT_SLICE})"
        ),
    )
    p.add_argument(
        "--reliability-csv",
        default=_DEFAULT_RELCSV,
        metavar="PATH",
        help=(
            "MetaQuantus reliability CSV to read "
            f"(default: {_DEFAULT_RELCSV})"
        ),
    )
    p.add_argument(
        "--output-csv",
        default=_DEFAULT_OUT,
        metavar="PATH",
        help=(
            "Destination for the wide-format ranking CSV "
            f"(default: {_DEFAULT_OUT})"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    slice_path  = Path(args.slice_csv)
    rel_path    = Path(args.reliability_csv)
    output_path = Path(args.output_csv)

    # 1. Load data
    raw = pd.read_csv(slice_path)
    # Reliability is optional: without it (or with all-NaN values),
    # metaquantus_discounted_weights falls back to the Autoweighted weights,
    # so the mqdiscount column equals autoweighted (a neutral, honest
    # placeholder until a clean meta-evaluation run is available).
    if rel_path.exists():
        meta_eval = pd.read_csv(rel_path)
    else:
        print(f"[compare_rankings] reliability CSV not found ({rel_path}); "
              "mqdiscount will fall back to autoweighted.")
        meta_eval = pd.DataFrame(
            columns=["model", "fae_method", "metric", "combined_reliability"]
        )

    # Filter to M* only
    raw_mstar = raw[raw["metric"].isin(M_STAR)].copy()

    # 2. Normalize
    normed = normalize_m_star(raw_mstar)

    # 3. Compute weights per model for autoweighted and mqdiscount
    aw_weights_all = autoweighted_weights(
        raw_mstar, metric_set=M_STAR, group_by=("model",)
    )
    mq_weights_all = metaquantus_discounted_weights(
        raw_mstar,
        reliability_df=meta_eval,
        metric_set=M_STAR,
        group_by=("model",),
        category_map=_CATEGORY_MAP,
    )

    # Uniform weights: 1/|M*| for each metric
    uniform_w = {m: 1.0 / len(M_STAR) for m in M_STAR}

    # Single-FC weights: 1.0 for fc, 0 for rest
    single_fc_w = {m: (1.0 if m == "faithfulness_correlation" else 0.0) for m in M_STAR}

    # 4. Aggregate per (model, image_id, fae_method)
    rows: list[dict] = []
    group_cols = ["model", "image_id", "fae_method"]

    for (model, image_id, fae_method), gdf in normed.groupby(group_cols):
        aw_w = aw_weights_all[(model,)]
        mq_w = mq_weights_all[(model,)]

        rows.append({
            "model": model,
            "fae_method": fae_method,
            "image_id": image_id,
            "effectiveness_uniform": aggregate_group(gdf, uniform_w, M_STAR),
            "effectiveness_autoweighted": aggregate_group(gdf, aw_w, M_STAR),
            "effectiveness_mqdiscount": aggregate_group(gdf, mq_w, M_STAR),
            "effectiveness_single_fc": aggregate_group(gdf, single_fc_w, M_STAR),
        })

    result = pd.DataFrame(rows)
    result = result[
        ["model", "fae_method", "image_id",
         "effectiveness_uniform", "effectiveness_autoweighted",
         "effectiveness_mqdiscount", "effectiveness_single_fc"]
    ]

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows → {output_path}\n")

    # -----------------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------------
    schemes = {
        "uniform": "effectiveness_uniform",
        "autoweighted": "effectiveness_autoweighted",
        "mqdiscount": "effectiveness_mqdiscount",
        "single_fc": "effectiveness_single_fc",
    }

    # Per-(model, fae) means
    means = (
        result.groupby(["model", "fae_method"])[list(schemes.values())]
        .mean()
        .reset_index()
        .sort_values("effectiveness_mqdiscount", ascending=False)
        .reset_index(drop=True)
    )
    means.index += 1

    # -----------------------------------------------------------------------
    # Table (i)
    # -----------------------------------------------------------------------
    print("```")
    print("## Table 1 — Per-(model, fae) mean effectiveness (sorted by mqdiscount ↓)")
    print()
    col_w = 26
    hdr = (
        f"{'model':<12}{'fae_method':<24}"
        f"{'uniform':>{col_w}}{'autoweighted':>{col_w}}"
        f"{'mqdiscount':>{col_w}}{'single_fc':>{col_w}}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, row in means.iterrows():
        print(
            f"{row['model']:<12}{row['fae_method']:<24}"
            f"{row['effectiveness_uniform']:>{col_w}.4f}"
            f"{row['effectiveness_autoweighted']:>{col_w}.4f}"
            f"{row['effectiveness_mqdiscount']:>{col_w}.4f}"
            f"{row['effectiveness_single_fc']:>{col_w}.4f}"
        )
    print()

    # -----------------------------------------------------------------------
    # Table (ii) — Spearman correlation matrix
    # -----------------------------------------------------------------------
    scheme_cols = list(schemes.values())
    scheme_labels = list(schemes.keys())
    corr_matrix = pd.DataFrame(
        np.eye(len(scheme_labels)), index=scheme_labels, columns=scheme_labels
    )

    for i, col_i in enumerate(scheme_cols):
        for j, col_j in enumerate(scheme_cols):
            if i != j:
                rho, _ = spearmanr(means[col_i], means[col_j])
                corr_matrix.iloc[i, j] = rho

    print("## Table 2 — Spearman ρ between scheme rankings (14 (model, fae) pairs)")
    print()
    w = 16
    print(f"{'':>{w}}" + "".join(f"{s:>{w}}" for s in scheme_labels))
    print("-" * (w * (len(scheme_labels) + 1)))
    for label in scheme_labels:
        row_str = f"{label:>{w}}"
        for col_label in scheme_labels:
            row_str += f"{corr_matrix.loc[label, col_label]:>{w}.4f}"
        print(row_str)
    print()

    # Flag divergent pairs
    divergent = []
    for i in range(len(scheme_labels)):
        for j in range(i + 1, len(scheme_labels)):
            rho = corr_matrix.iloc[i, j]
            if abs(rho) < 0.85:
                divergent.append(
                    f"  {scheme_labels[i]} vs {scheme_labels[j]}: ρ={rho:.4f} "
                    f"← |ρ| < 0.85, meaningful divergence"
                )
    if divergent:
        print("Pairs with |ρ| < 0.85:")
        print("\n".join(divergent))
    else:
        print("All pairs: |ρ| ≥ 0.85 — no meaningful divergence between schemes.")
    print()

    # -----------------------------------------------------------------------
    # Table (iii) — Rank shifts uniform → mqdiscount
    # -----------------------------------------------------------------------
    means["rank_uniform"] = means["effectiveness_uniform"].rank(
        ascending=False, method="min"
    ).astype(int)
    means["rank_mqdiscount"] = means["effectiveness_mqdiscount"].rank(
        ascending=False, method="min"
    ).astype(int)
    means["rank_shift"] = means["rank_mqdiscount"] - means["rank_uniform"]
    means["abs_shift"] = means["rank_shift"].abs()

    top5 = means.nlargest(5, "abs_shift")[
        ["model", "fae_method", "rank_uniform", "rank_mqdiscount", "rank_shift"]
    ].reset_index(drop=True)
    top5.index += 1

    print("## Table 3 — Top-5 rank shifts: uniform → mqdiscount")
    print("  (rank_shift = rank_mqdiscount − rank_uniform; negative = moved up)")
    print()
    hdr3 = (
        f"{'model':<12}{'fae_method':<24}"
        f"{'rank_uniform':>14}{'rank_mqdiscount':>16}{'rank_shift':>12}"
    )
    print(hdr3)
    print("-" * len(hdr3))
    for _, row in top5.iterrows():
        shift = int(row["rank_shift"])
        arrow = f"{shift:+d}"
        print(
            f"{row['model']:<12}{row['fae_method']:<24}"
            f"{int(row['rank_uniform']):>14}{int(row['rank_mqdiscount']):>16}"
            f"{arrow:>12}"
        )
    print("```")


if __name__ == "__main__":
    main()
