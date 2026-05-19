"""
Metric selection: combines redundancy pruning and meta-validation to produce M*.

Responsibilities (see docs/thesis_plan.md §6, Meta-Evaluation):
    - Accept the correlation matrix from redundancy.py and the meta-validation
      scores from metaquantus_wrapper.py
    - Apply the two-stage filter: (1) discard redundant metrics per category,
      (2) flag metrics with low NR or AR scores for reporting
    - Return the final validated metric set M* used by the aggregation layer

Not responsible for: computing correlation or meta-validation scores directly
(those are delegated to redundancy.py and metaquantus_wrapper.py),
or the aggregation of FAE method scores (see aggregation/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Axiomatic metrics and their expected value when the axiom is satisfied.
# completeness: sum of attributions equals f(x) - f(baseline) → deviation = 0.
# non_sensitivity: attribution of constant features = 0 → deviation = 0.
_AXIOMATIC_EXPECTED: dict[str, float] = {
    "completeness": 0.0,
    "non_sensitivity": 0.0,
}


def verify_axiomatic_properties(
    df: pd.DataFrame,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Pre-flight check: verify axiomatic metric scores match their expected values.

    For each (model, fae_method) combination, computes the mean absolute
    deviation of the metric score from the expected value when the axiom is
    satisfied. This check provides the intellectual basis for excluding
    axiomatic metrics from aggregation: if the axiom holds (mean_violation ≈ 0),
    the metric contributes no discriminative signal and should be treated as a
    binary pass/fail indicator rather than a continuous effectiveness score.

    Only axiomatic metrics with at least one non-NaN observation are included.
    FAE methods for which the metric is entirely NaN (e.g., Completeness for
    GradCAM, which does not satisfy the completeness axiom) are excluded from
    the output — their NaN scores are a design constraint, not a violation.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns: model, image_id, fae_method,
        metric, score. Typically ``results/vertical_slice_7fae_12metrics.csv``.
    tolerance : float
        Maximum mean absolute deviation from the expected value for a
        (model, fae_method, axiom) triplet to be considered *passing*.
        Default ``1e-4``.

    Returns
    -------
    pd.DataFrame
        Columns: model, fae_method, axiom_name, mean_violation,
        n_observations, passes (bool).
        One row per (model, fae_method, axiom_name) with at least one
        non-NaN score. Sorted by axiom_name, model, fae_method.
    """
    axiomatic_df = df[
        df["metric"].isin(_AXIOMATIC_EXPECTED) & df["score"].notna()
    ].copy()

    rows = []
    for (model, fae, axiom), group in axiomatic_df.groupby(
        ["model", "fae_method", "metric"]
    ):
        expected = _AXIOMATIC_EXPECTED[axiom]
        violations = (group["score"] - expected).abs()
        mean_viol = float(violations.mean())
        n_obs = len(violations)
        rows.append(
            {
                "model": model,
                "fae_method": fae,
                "axiom_name": axiom,
                "mean_violation": mean_viol,
                "n_observations": n_obs,
                "passes": mean_viol < tolerance,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model", "fae_method", "axiom_name",
                "mean_violation", "n_observations", "passes",
            ]
        )

    result = pd.DataFrame(rows).sort_values(
        ["axiom_name", "model", "fae_method"]
    ).reset_index(drop=True)
    return result
