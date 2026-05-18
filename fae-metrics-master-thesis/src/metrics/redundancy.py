"""
Metric redundancy analysis — Research Contribution 1.

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 1;
and §10, Decision D4):
    - Compute a Spearman correlation matrix across all metrics for a given
      model × dataset × FAE-method combination
    - Apply the |ρ| > 0.85 (default, Decision D4) pruning threshold within
      each Quantus category to identify a non-redundant representative subset M*
    - Return the reduced metric set and the full correlation matrix for
      reporting in Chapter 4

Not responsible for: computing the metric scores themselves (see quantus_wrapper.py),
meta-validation weighting (see meta_evaluation/),
or score aggregation (see aggregation/).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Canonical category assignment for all 12 Quantus metrics used in the
# vertical slice (Hedström et al., 2023 taxonomy).
METRIC_CATEGORIES: dict[str, str] = {
    "faithfulness_correlation": "Faithfulness",
    "pixel_flipping": "Faithfulness",
    "max_sensitivity": "Robustness",
    "avg_sensitivity": "Robustness",
    "relevance_mass_accuracy": "Localization",
    "pointing_game": "Localization",
    "sparseness": "Complexity",
    "complexity": "Complexity",
    "model_parameter_randomisation": "Randomization",
    "random_logit": "Randomization",
    "completeness": "Axiomatic",
    "non_sensitivity": "Axiomatic",
}


def compute_redundancy_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 30,
    group_by: tuple[str, ...] = ("model",),
) -> dict[tuple, pd.DataFrame]:
    """Compute pairwise metric correlation matrices, one per group.

    Pivots the long-format DataFrame to wide format (rows = image_id ×
    fae_method, columns = metrics) within each group, then computes
    pairwise Spearman or Pearson correlations using pandas pairwise
    deletion for NaN entries.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns: model, image_id, fae_method,
        metric, score.
    method : {'spearman', 'pearson'}
        Correlation method. Default ``'spearman'`` (rank-based, more
        robust to outliers and non-linear metric relationships).
    min_obs : int
        Minimum number of non-NaN observations required to include a
        metric in the correlation matrix. Metrics with fewer valid rows
        (e.g., ``non_sensitivity`` with all-NaN) are dropped silently.
    group_by : tuple of str
        Columns that define independent groups; each group produces one
        correlation matrix.

    Returns
    -------
    dict[tuple, pd.DataFrame]
        Mapping from group-key tuple to square symmetric correlation
        DataFrame. Row and column labels are metric names. Empty
        DataFrame if fewer than two metrics survive the ``min_obs``
        filter.

    Raises
    ------
    ValueError
        If *method* is not ``'spearman'`` or ``'pearson'``.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")

    result: dict[tuple, pd.DataFrame] = {}

    for group_keys, group_df in df.groupby(list(group_by)):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        wide = group_df.pivot_table(
            index=["image_id", "fae_method"],
            columns="metric",
            values="score",
            aggfunc="first",
        )
        wide.columns.name = None

        valid_counts = wide.notna().sum()
        wide = wide.loc[:, valid_counts >= min_obs]

        if wide.shape[1] < 2:
            result[group_keys] = pd.DataFrame()
            continue

        corr_matrix = wide.corr(method=method)
        result[group_keys] = corr_matrix

    return result


def prune_redundant_metrics(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.85,
    metric_directions: Optional[dict[str, int]] = None,  # noqa: ARG001
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy pruning of redundant metrics at |ρ| > threshold.

    Iteratively drops the metric with the highest mean absolute correlation
    to all currently surviving metrics, until no pair exceeds the threshold.
    Tie-breaking favours removing the metric that is more correlated on
    average with the rest of the surviving set.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square symmetric correlation matrix from :func:`compute_redundancy_matrix`.
    threshold : float
        Absolute correlation threshold; pairs with |ρ| > threshold are
        considered redundant. Default ``0.85`` per Decision D4.
    metric_directions : dict[str, int] or None
        Reserved for future direction-aware tie-breaking. Not used in
        the current implementation.

    Returns
    -------
    kept : list[str]
        Non-redundant metric names in their original column order.
    pruned_pairs : list[tuple[str, str, float]]
        One entry per removal: ``(dropped, retained, |ρ|)``.
    """
    if corr_matrix.empty:
        return [], []

    original_order = list(corr_matrix.columns)
    surviving = set(original_order)
    pruned_pairs: list[tuple[str, str, float]] = []

    changed = True
    while changed:
        changed = False
        sub = corr_matrix.loc[list(surviving), list(surviving)]
        abs_corr = sub.abs()

        cols = list(sub.columns)
        for i, m1 in enumerate(cols):
            for m2 in cols[i + 1:]:
                rho = abs_corr.at[m1, m2]
                if rho > threshold:
                    mean1 = abs_corr[m1].drop(m1).mean()
                    mean2 = abs_corr[m2].drop(m2).mean()
                    drop, keep = (m1, m2) if mean1 >= mean2 else (m2, m1)
                    surviving.discard(drop)
                    pruned_pairs.append((drop, keep, float(rho)))
                    changed = True
                    break
            if changed:
                break

    kept = [m for m in original_order if m in surviving]
    return kept, pruned_pairs


def category_redundancy_summary(
    corr_matrix: pd.DataFrame,
    metric_categories: dict[str, str],
) -> pd.DataFrame:
    """Summarise within-category mean absolute correlation.

    For each category present in *corr_matrix*, computes the mean of all
    off-diagonal |ρ| values within that category.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix from :func:`compute_redundancy_matrix`.
    metric_categories : dict[str, str]
        Mapping from metric name to category name (e.g. :data:`METRIC_CATEGORIES`).

    Returns
    -------
    pd.DataFrame
        Indexed by category name with columns:

        - ``n_metrics``: metrics from this category present in the matrix.
        - ``mean_abs_rho``: mean off-diagonal |ρ| within category.
          ``NaN`` when the category has fewer than 2 metrics in the matrix.
    """
    if corr_matrix.empty:
        return pd.DataFrame(columns=["n_metrics", "mean_abs_rho"])

    metrics_in_matrix = set(corr_matrix.columns)
    cat_members: dict[str, list[str]] = {}
    for metric, cat in metric_categories.items():
        if metric in metrics_in_matrix:
            cat_members.setdefault(cat, []).append(metric)

    rows = []
    for cat in sorted(cat_members):
        members = cat_members[cat]
        n = len(members)
        if n < 2:
            mean_rho = float("nan")
        else:
            sub = corr_matrix.loc[members, members].abs()
            mask = ~np.eye(n, dtype=bool)
            mean_rho = float(sub.values[mask].mean())
        rows.append({"category": cat, "n_metrics": n, "mean_abs_rho": mean_rho})

    return pd.DataFrame(rows).set_index("category")
