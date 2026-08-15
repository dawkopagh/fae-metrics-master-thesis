"""
Per-metric normalisation of raw quantus scores before aggregation.

Responsibilities (see docs/thesis_plan.md §6, Aggregation; Decision D1):
    - Implement Second Moment Scaling (default, Hryniewska-Guzik et al., 2024):
      s̃ = s / sqrt(E[s²])
    - Implement z-score and min-max normalisation for ablation comparisons
    - Apply direction rectification so that higher score always means better
      for all metrics (some quantus metrics are lower-is-better by convention)
    - Accept a long-format DataFrame of raw scores; return the same schema
      with a normalised_score column appended

Not responsible for: computing raw scores (see metrics/), weighting
(see weighting.py), or the final aggregation operator (see aggregate.py).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric direction registry
# ---------------------------------------------------------------------------
# +1 = higher raw score means better explanation quality
# -1 = lower raw score means better explanation quality
#
# Expand this registry whenever a new metric is introduced upstream.
# See docs/thesis_plan.md §5 for planned additions.
METRIC_DIRECTIONS: dict[str, int] = {
    # Faithfulness
    "faithfulness_correlation": +1,  # higher = better
    # Quantus PixelFlipping (return_auc_per_sample=True) returns the AUC of
    # the prediction curve on progressively perturbed inputs (MoRF order):
    # a faithful attribution makes the prediction collapse early, so LOWER
    # AUC = better. Matches quantus ScoreDirection.LOWER. Was erroneously +1
    # until 2026-08-15, which inverted this metric inside every aggregate.
    "pixel_flipping": -1,  # AUC of degradation curve, lower = better
    # Robustness
    "max_sensitivity": -1,  # lower = better
    "avg_sensitivity": -1,  # lower = better
    # Localization
    "relevance_mass_accuracy": +1,  # higher = better
    "pointing_game": +1,  # higher = better
    # Complexity
    "sparseness": +1,  # higher = better (Gini-like)
    "complexity": -1,  # lower = better (entropy)
    # Randomization
    "model_parameter_randomisation": -1,  # lower = better
    "random_logit": -1,  # lower = better
    # Axiomatic
    # Quantus Completeness returns a boolean (1.0 = axiom satisfied), so
    # higher = better (quantus ScoreDirection.HIGHER). Direction is moot in
    # practice: the score is constant per method and pre-screened out of M*.
    "completeness": +1,  # bool axiom-satisfied, higher = better
    "non_sensitivity": -1,  # lower = better
}


def _resolve_direction(metric: str) -> int:
    """Look up the canonical direction for *metric*.

    Parameters
    ----------
    metric : str
        Metric name as it appears in the ``metric`` column.

    Returns
    -------
    int
        ``+1`` or ``-1``.

    Raises
    ------
    KeyError
        If *metric* is not in :data:`METRIC_DIRECTIONS`.
    """
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError:
        raise KeyError(
            f"Unknown metric '{metric}'. Register it in "
            f"METRIC_DIRECTIONS before normalizing. "
            f"Known metrics: {sorted(METRIC_DIRECTIONS)}"
        ) from None


# ---------------------------------------------------------------------------
# Element-wise normalization strategies
# ---------------------------------------------------------------------------

def second_moment_scale(scores: np.ndarray | pd.Series,
                        direction: int) -> np.ndarray:
    r"""Second Moment Scaling (Hryniewska-Guzik et al., 2024).

    .. math::
        \tilde{s}_i = \frac{s_i}{\sqrt{\frac{1}{n}\sum_j s_j^2}}
        \cdot \mathrm{direction}

    The denominator is the root-mean-square of the raw scores, which
    preserves the sign structure while mapping the "typical" magnitude
    to ±1. Multiplication by *direction* converts lower-is-better
    metrics to higher-is-better orientation.

    Parameters
    ----------
    scores : array-like
        1-D raw metric scores (one metric, many samples).
    direction : int
        ``+1`` (higher-is-better) or ``-1`` (lower-is-better).

    Returns
    -------
    np.ndarray
        Normalised scores. If all input scores are zero, returns zeros.
    """
    s = np.asarray(scores, dtype=np.float64)
    rms = np.sqrt(np.mean(s ** 2))
    if rms == 0.0:
        return np.zeros_like(s)
    return (s / rms) * direction


def z_score(scores: np.ndarray | pd.Series,
            direction: int) -> np.ndarray:
    r"""Standard z-score normalisation with direction rectification.

    .. math::
        \tilde{s}_i = \frac{s_i - \bar{s}}{\sigma_s} \cdot \mathrm{direction}

    Parameters
    ----------
    scores : array-like
        1-D raw metric scores.
    direction : int
        ``+1`` or ``-1``.

    Returns
    -------
    np.ndarray
        Normalised scores. Returns zeros when :math:`\sigma = 0` (all
        scores identical).
    """
    s = np.asarray(scores, dtype=np.float64)
    std = np.std(s, ddof=0)
    if std == 0.0:
        return np.zeros_like(s)
    return ((s - np.mean(s)) / std) * direction


def min_max(scores: np.ndarray | pd.Series,
            direction: int) -> np.ndarray:
    r"""Min-max normalisation to [0, 1] with direction rectification.

    For direction ``+1`` (higher-is-better):

    .. math::
        \tilde{s}_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}

    For direction ``-1`` (lower-is-better), the result is flipped:

    .. math::
        \tilde{s}_i = 1 - \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}

    This ensures that higher normalised score always means *better*.

    Parameters
    ----------
    scores : array-like
        1-D raw metric scores.
    direction : int
        ``+1`` or ``-1``.

    Returns
    -------
    np.ndarray
        Normalised scores in [0, 1]. Returns 0.5 for all samples when
        ``max == min`` (constant input).
    """
    s = np.asarray(scores, dtype=np.float64)
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return np.full_like(s, 0.5)
    normed = (s - s_min) / (s_max - s_min)
    if direction == -1:
        normed = 1.0 - normed
    return normed


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------
_NORMALIZERS = {
    "second_moment": second_moment_scale,
    "z_score": z_score,
    "min_max": min_max,
}


# ---------------------------------------------------------------------------
# DataFrame-level interface
# ---------------------------------------------------------------------------

def normalize_scores(
    df: pd.DataFrame,
    method: Literal["second_moment", "z_score", "min_max"] = "second_moment",
    group_by: tuple[str, ...] = ("model", "metric"),
) -> pd.DataFrame:
    """Normalise raw metric scores within groups, adding ``score_normalized``.

    Normalisation is applied **within** each group defined by *group_by*.
    The default ``('model', 'metric')`` ensures each metric's scores are
    normalised separately per model, preventing one model's saturated
    metric from dominating another's.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with at least ``metric`` and ``score``
        columns, plus every column listed in *group_by*.
    method : {'second_moment', 'z_score', 'min_max'}
        Normalisation strategy. Default ``'second_moment'`` per Decision D1.
    group_by : tuple of str
        Columns that define independent normalisation groups.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``score_normalized`` column.
        The original ``score`` column is retained unchanged.

    Raises
    ------
    KeyError
        If an unknown metric is encountered (not in :data:`METRIC_DIRECTIONS`).
    ValueError
        If *method* is not one of the supported strategies.
    """
    if method not in _NORMALIZERS:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Choose from {sorted(_NORMALIZERS)}."
        )
    normalizer = _NORMALIZERS[method]

    result = df.copy()
    result["score_normalized"] = np.nan

    for group_keys, idx in result.groupby(list(group_by)).groups.items():
        # group_keys is a tuple when group_by has >1 column, scalar otherwise
        if isinstance(group_keys, str):
            group_keys = (group_keys,)

        # Identify metric name from the group — find the 'metric' column value
        group_dict = dict(zip(group_by, group_keys))
        metric_name = group_dict.get("metric")
        if metric_name is None:
            # metric not in group_by — all rows in group may span metrics.
            # Fall back to getting unique metric from the subset.
            unique_metrics = result.loc[idx, "metric"].unique()
            if len(unique_metrics) != 1:
                raise ValueError(
                    "group_by does not include 'metric', yet rows in a single "
                    "group span multiple metrics. Include 'metric' in group_by "
                    "or ensure each group contains exactly one metric."
                )
            metric_name = unique_metrics[0]

        direction = _resolve_direction(metric_name)
        raw = result.loc[idx, "score"].values
        result.loc[idx, "score_normalized"] = normalizer(raw, direction)

    return result
