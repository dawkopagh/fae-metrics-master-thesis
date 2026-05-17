"""
Aggregation operators that combine normalised metric scores into a scalar
effectiveness index E(Φ) per FAE method per model.

Responsibilities (see docs/thesis_plan.md §6, Aggregation; Decision D2):
    - Implement weighted mean aggregation (default, recommended in thesis)
    - Implement min aggregation and geometric mean aggregation for comparison
    - Accept normalised scores and per-metric weights; return one E(Φ) scalar
      per (model, fae_method) pair
    - Produce a results DataFrame with all three operator scores for reporting

Not responsible for: normalisation (see normalize.py),
weight computation (see weighting.py),
or statistical comparison across FAE methods (see comparison/).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar aggregation operators
# ---------------------------------------------------------------------------

def weighted_mean(normalized: np.ndarray,
                  weights: np.ndarray | None = None) -> float:
    r"""Weighted arithmetic mean of normalised scores, skipping NaN entries.

    .. math::
        E = \sum_{k \in \mathcal{V}} w_k' \, \tilde{s}_k

    where :math:`\mathcal{V}` is the set of non-NaN indices and
    :math:`w_k' = w_k / \sum_{j \in \mathcal{V}} w_j` are the renormalised
    weights over valid entries. If *weights* is ``None``, uniform weights are
    used before renormalisation (equivalent to ``np.nanmean``). If **all**
    inputs are NaN, returns ``float('nan')``.

    Parameters
    ----------
    normalized : np.ndarray
        1-D array of normalised metric scores for one group.
    weights : np.ndarray or None
        Per-metric weights summing to 1. If ``None``, uniform weights
        :math:`w_k = 1/K` are used. Weights are renormalised over non-NaN
        entries, so the original weights need only sum to 1 over all K entries.

    Returns
    -------
    float
        Aggregated effectiveness index, or ``float('nan')`` if all inputs
        are NaN.

    Raises
    ------
    ValueError
        If *weights* do not sum to 1.0 (tolerance 1e-6).
    """
    s = np.asarray(normalized, dtype=np.float64)
    if weights is None:
        if np.all(np.isnan(s)):
            return float("nan")
        return float(np.nanmean(s))
    w = np.asarray(weights, dtype=np.float64)
    if abs(w.sum() - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {w.sum():.8f}."
        )
    valid = ~np.isnan(s)
    if not valid.any():
        return float("nan")
    w_valid = w[valid]
    w_renorm = w_valid / w_valid.sum()
    return float(np.dot(w_renorm, s[valid]))


def minimum(normalized: np.ndarray,
            weights: np.ndarray | None = None) -> float:
    r"""Minimum (worst-case) aggregation, skipping NaN entries.

    .. math::
        E = \min_{k \in \mathcal{V}} \tilde{s}_k

    where :math:`\mathcal{V}` is the set of non-NaN indices. Returns
    ``float('nan')`` if all inputs are NaN.

    The *weights* parameter is accepted for API uniformity with
    :func:`weighted_mean` but is ignored: the minimum is an
    order-statistical operator and weighting does not apply.

    Parameters
    ----------
    normalized : np.ndarray
        1-D array of normalised scores.
    weights : np.ndarray or None
        Ignored. Present for API uniformity.

    Returns
    -------
    float
        The smallest non-NaN normalised score in the group, or
        ``float('nan')`` if all are NaN.
    """
    s = np.asarray(normalized, dtype=np.float64)
    if np.all(np.isnan(s)):
        return float("nan")
    return float(np.nanmin(s))


def geometric_mean(normalized: np.ndarray,
                   weights: np.ndarray | None = None) -> float:
    r"""Weighted geometric mean of normalised scores, skipping NaN entries.

    .. math::
        E = \exp\!\Bigl(\sum_{k \in \mathcal{V}} w_k' \ln(\tilde{s}_k + \epsilon)\Bigr)

    where :math:`\mathcal{V}` is the set of non-NaN indices,
    :math:`w_k' = w_k / \sum_{j \in \mathcal{V}} w_j` are renormalised weights,
    and :math:`\epsilon = 10^{-12}` avoids :math:`\ln(0)`. Returns
    ``float('nan')`` if all inputs are NaN.

    Parameters
    ----------
    normalized : np.ndarray
        1-D array of normalised scores. Non-NaN entries must be ≥ 0.
    weights : np.ndarray or None
        Per-metric weights summing to 1. ``None`` → uniform. Weights are
        renormalised over non-NaN entries.

    Returns
    -------
    float
        Aggregated effectiveness index, or ``float('nan')`` if all inputs
        are NaN.

    Raises
    ------
    ValueError
        If any non-NaN element of *normalized* is negative.
    """
    s = np.asarray(normalized, dtype=np.float64)
    valid = ~np.isnan(s)
    if not valid.any():
        return float("nan")
    s_valid = s[valid]
    if np.any(s_valid < 0):
        raise ValueError(
            "geometric_mean requires all normalised scores >= 0. "
            f"Got min = {s_valid.min():.6f}."
        )
    eps = 1e-12
    if weights is None:
        w_valid = np.full(valid.sum(), 1.0 / valid.sum())
    else:
        w = np.asarray(weights, dtype=np.float64)
        w_sub = w[valid]
        w_valid = w_sub / w_sub.sum()
    return float(np.exp(np.sum(w_valid * np.log(s_valid + eps))))


# ---------------------------------------------------------------------------
# Operator dispatcher
# ---------------------------------------------------------------------------
_OPERATORS: dict[str, callable] = {
    "weighted_mean": weighted_mean,
    "min": minimum,
    "geometric_mean": geometric_mean,
}


# ---------------------------------------------------------------------------
# DataFrame-level interface
# ---------------------------------------------------------------------------

def aggregate_scores(
    df: pd.DataFrame,
    operator: Literal["weighted_mean", "min", "geometric_mean"] = "weighted_mean",
    weights: dict[str, float] | None = None,
    group_by: tuple[str, ...] = ("model", "image_id", "fae_method"),
) -> pd.DataFrame:
    """Aggregate normalised metric scores into one effectiveness index per group.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame that **must** contain a ``score_normalized``
        column (output of :func:`normalize_scores`), a ``metric`` column,
        and every column listed in *group_by*.
    operator : {'weighted_mean', 'min', 'geometric_mean'}
        Aggregation strategy. Default ``'weighted_mean'`` per Decision D2.
    weights : dict mapping metric name → float, or None
        Per-metric weights. If ``None``, uniform weights are used.
        When provided, weights must sum to 1.0 (tolerance 1e-6).
    group_by : tuple of str
        Columns that define the aggregation groups. Default produces one
        effectiveness index per (model, image, FAE method).

    Returns
    -------
    pd.DataFrame
        Columns: ``*group_by, effectiveness_index, operator``.
        One row per group.

    Raises
    ------
    ValueError
        If ``score_normalized`` column is missing from *df*, or if
        *operator* is unknown.
    """
    if "score_normalized" not in df.columns:
        raise ValueError(
            "Column 'score_normalized' not found in DataFrame. "
            "Run normalize_scores() first."
        )
    if operator not in _OPERATORS:
        raise ValueError(
            f"Unknown operator '{operator}'. "
            f"Choose from {sorted(_OPERATORS)}."
        )

    op_fn = _OPERATORS[operator]
    rows: list[dict] = []

    for group_keys, group_df in df.groupby(list(group_by)):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        metrics_in_group = group_df["metric"].values
        scores = group_df["score_normalized"].values

        # Build weight vector aligned with scores
        w: np.ndarray | None = None
        if weights is not None:
            w_list = [weights[m] for m in metrics_in_group]
            w = np.array(w_list, dtype=np.float64)

        eff = op_fn(scores, w)
        row = dict(zip(group_by, group_keys))
        row["effectiveness_index"] = eff
        row["operator"] = operator
        rows.append(row)

    return pd.DataFrame(rows)
