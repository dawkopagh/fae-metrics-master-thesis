"""
Weight computation for the aggregation layer — Research Contribution 2 (novel part).

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 2;
§6, Aggregation; Decision D3):
    - Implement Autoweighted weighting (NormEnsembleXAI, Hryniewska-Guzik 2024):
      weights inversely proportional to coefficient of variation of each metric's
      scores across the test set.
    - Implement the MetaQuantus-discounted extension: multiply Autoweighted
      coefficients by the NR × AR meta-validation reliability for each metric
      (novel contribution of the thesis — Part B, deferred pending
      results/meta_evaluation_reliability.csv).
    - Return per-metric weight vectors w ∈ ℝ^|M*| keyed by group tuple.

Not responsible for: normalisation (see normalize.py),
aggregation itself (see aggregate.py),
or computing meta-validation scores (see meta_evaluation/).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-8


def autoweighted_weights(
    scores_df: pd.DataFrame,
    metric_set: list[str] | None = None,
    group_by: tuple = ("model",),
) -> dict[tuple, dict[str, float]]:
    """Compute Autoweighted weights per group.

    For each group (default: per model) and for each metric in *metric_set*:

    .. math::
        \\mathrm{cv}_k = \\frac{\\mathrm{std}(s_k)}{|\\mathrm{mean}(s_k)|}
        \\qquad
        r_k = \\frac{1}{\\mathrm{cv}_k + \\varepsilon}
        \\qquad
        w_k = \\frac{r_k}{\\sum_j r_j}

    where :math:`\\varepsilon = 10^{-8}` provides numerical stability and
    all statistics are NaN-aware (``np.nanstd``, ``np.nanmean``).

    Autoweighted gives higher weight to metrics with low coefficient of
    variation, i.e. metrics whose scores are internally consistent across
    the test set.  Metrics that are noisy (high CV relative to their mean)
    are penalised.  Metrics with all-NaN scores in a group receive
    ``autoweighted_k = 0`` and the remaining weights renormalise to sum to 1.

    Reference: Hryniewska-Guzik et al. (2024), NormEnsembleXAI.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Long-format DataFrame with at least columns ``metric`` and ``score``,
        plus every column listed in *group_by*.  Raw (pre-normalisation)
        metric scores are expected; CV is scale-invariant so normalised scores
        produce the same result.
    metric_set : list[str] or None
        Metrics to include.  If ``None``, uses every unique metric present in
        *scores_df*.
    group_by : tuple of str
        Columns that define independent weighting groups.  Default
        ``('model',)`` — one weight vector per model.

    Returns
    -------
    dict[tuple, dict[str, float]]
        ``{group_key_tuple: {metric_name: weight}}``.  Weights within each
        group sum to 1.0 (up to floating-point rounding), except the
        degenerate case where every metric is all-NaN (uniform fallback).
    """
    if metric_set is None:
        metric_set = sorted(scores_df["metric"].unique().tolist())

    result: dict[tuple, dict[str, float]] = {}

    for group_keys, group_df in scores_df.groupby(list(group_by)):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        raw_weights: dict[str, float] = {}

        for metric in metric_set:
            vals = group_df.loc[group_df["metric"] == metric, "score"].to_numpy(
                dtype=float
            )

            if len(vals) == 0 or np.all(np.isnan(vals)):
                raw_weights[metric] = 0.0
                continue

            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)

            # Guard: if mean ≈ 0, CV would be infinite; treat denominator as ε
            # so the metric is regarded as very noisy (tiny weight).
            denom = abs(mean_val) if abs(mean_val) >= _EPS else _EPS
            cv_k = std_val / denom
            raw_weights[metric] = 1.0 / (cv_k + _EPS)

        total = sum(raw_weights.values())

        if total <= 0.0:
            # Degenerate: every metric is all-NaN in this group.
            n = len(metric_set)
            norm_weights: dict[str, float] = {
                m: (1.0 / n if n > 0 else 0.0) for m in metric_set
            }
        else:
            norm_weights = {m: w / total for m, w in raw_weights.items()}

        result[group_keys] = norm_weights

    return result


def metaquantus_discounted_weights(
    scores_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    metric_set: list[str] | None = None,
    group_by: tuple = ("model",),
    reliability_column: str = "combined_reliability",
    category_map: dict[str, str] | None = None,
    return_sources: bool = False,
) -> dict[tuple, dict[str, float]] | tuple:
    """Compute the novel weighting: Autoweighted × MetaQuantus reliability discount.

    For each group (per-model) and each metric k:

    .. math::
        w_k = \\frac{r_k^{\\mathrm{AW}} \\cdot \\lambda_k}{
                \\sum_j r_j^{\\mathrm{AW}} \\cdot \\lambda_j}

    where :math:`r_k^{\\mathrm{AW}}` comes from :func:`autoweighted_weights`
    and :math:`\\lambda_k` is the reliability factor resolved via the fallback
    hierarchy below.

    **Fallback hierarchy** (applied when *combined_reliability* is NaN):

    1. **NR-only**: if ``nr_score`` column exists and is finite, use it.
       Applies to metrics that internally re-compute attributions
       (avg/max_sensitivity, MPR, random_logit); AR is methodologically
       inapplicable — NR-only reliability is the principled choice.
    2. **Category mean**: if *category_map* is provided, use the mean
       reliability of other valid metrics in the same category
       (``combined_reliability`` where finite, ``nr_score`` otherwise).
    3. **Global mean**: mean of all finite ``combined_reliability`` values
       across the current group.
    4. **No discount**: 1.0 when no valid reliability exists anywhere.

    Each fallback step taken is logged at INFO level.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Long-format scores (same schema as :func:`autoweighted_weights`).
    reliability_df : pd.DataFrame
        MetaQuantus output with columns: model, fae_method, metric,
        combined_reliability (and optionally nr_score).  Aggregated to
        per-(model, metric) by nanmean across fae_method before applying.
    metric_set : list[str] or None
        Metrics to include.  Defaults to all unique metrics in *scores_df*.
    group_by : tuple of str
        Columns for independent weighting groups.
    reliability_column : str
        Column in *reliability_df* carrying the primary reliability score.
    category_map : dict[str, str] or None
        Mapping from metric name to Quantus category.  Required for fallback 2.
    return_sources : bool
        If ``True``, return ``(weights_dict, sources_dict)`` where
        *sources_dict* maps ``{group_key: {metric: source_label}}``.
        Source labels: ``'combined'``, ``'nr_only'``, ``'category_fallback'``,
        ``'global_fallback'``, ``'no_discount'``.

    Returns
    -------
    dict[tuple, dict[str, float]]
        ``{group_key_tuple: {metric_name: weight}}``, or a 2-tuple when
        *return_sources* is ``True``.
    """
    if metric_set is None:
        metric_set = sorted(scores_df["metric"].unique().tolist())

    aw_result = autoweighted_weights(scores_df, metric_set=metric_set, group_by=group_by)
    has_nr = "nr_score" in reliability_df.columns

    all_weights: dict[tuple, dict[str, float]] = {}
    all_sources: dict[tuple, dict[str, str]] = {}

    for group_keys, aw_w in aw_result.items():
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        # Filter reliability rows to this group.
        group_rel = reliability_df
        if "model" in reliability_df.columns and "model" in group_by:
            model_idx = list(group_by).index("model")
            group_rel = reliability_df[
                reliability_df["model"] == group_keys[model_idx]
            ]

        # Aggregate across fae_method by nanmean per metric.
        agg_cols = [reliability_column] + (["nr_score"] if has_nr else [])
        if "fae_method" in group_rel.columns:
            rel_per_metric = group_rel.groupby("metric")[agg_cols].mean()
        else:
            rel_per_metric = group_rel.set_index("metric")[agg_cols]

        # Pre-compute global mean of finite combined_reliability (fallback 3).
        global_finite = [
            float(rel_per_metric.loc[m, reliability_column])
            for m in metric_set
            if m in rel_per_metric.index
            and pd.notna(rel_per_metric.loc[m, reliability_column])
        ]
        global_mean: float | None = float(np.mean(global_finite)) if global_finite else None

        # Resolve reliability factor λ_k per metric.
        lambda_k: dict[str, float] = {}
        sources: dict[str, str] = {}

        for m in metric_set:
            in_index = m in rel_per_metric.index
            combined = (
                float(rel_per_metric.loc[m, reliability_column])
                if in_index else float("nan")
            )
            nr = (
                float(rel_per_metric.loc[m, "nr_score"])
                if (has_nr and in_index) else float("nan")
            )
            # Ensure NaN sentinel (pandas NaN → Python float nan).
            if not pd.notna(combined):
                combined = float("nan")
            if not pd.notna(nr):
                nr = float("nan")

            if pd.notna(combined):
                lambda_k[m] = combined
                sources[m] = "combined"

            elif has_nr and pd.notna(nr):
                # Step 1: NR-only — AR inapplicable for this metric.
                lambda_k[m] = nr
                sources[m] = "nr_only"
                logger.info(
                    "group=%s metric=%s: AR inapplicable; NR-only reliability=%.4f",
                    group_keys, m, nr,
                )

            elif category_map is not None and category_map.get(m) is not None:
                # Step 2: category mean.
                cat = category_map[m]
                peer_vals: list[float] = []
                for peer in metric_set:
                    if peer == m or category_map.get(peer) != cat:
                        continue
                    if peer not in rel_per_metric.index:
                        continue
                    p_cr = float(rel_per_metric.loc[peer, reliability_column])
                    p_nr = (
                        float(rel_per_metric.loc[peer, "nr_score"])
                        if has_nr else float("nan")
                    )
                    if not pd.notna(p_cr):
                        p_cr = float("nan")
                    if not pd.notna(p_nr):
                        p_nr = float("nan")
                    if pd.notna(p_cr):
                        peer_vals.append(p_cr)
                    elif has_nr and pd.notna(p_nr):
                        peer_vals.append(p_nr)
                if peer_vals:
                    lambda_k[m] = float(np.mean(peer_vals))
                    sources[m] = "category_fallback"
                    logger.info(
                        "group=%s metric=%s: NaN reliability; category '%s' "
                        "fallback → λ=%.4f",
                        group_keys, m, cat, lambda_k[m],
                    )

            if m not in lambda_k:
                if global_mean is not None:
                    # Step 3: global mean.
                    lambda_k[m] = global_mean
                    sources[m] = "global_fallback"
                    logger.info(
                        "group=%s metric=%s: NaN reliability; global mean "
                        "fallback → λ=%.4f",
                        group_keys, m, global_mean,
                    )
                else:
                    # Step 4: no discount.
                    lambda_k[m] = 1.0
                    sources[m] = "no_discount"
                    logger.info(
                        "group=%s metric=%s: no valid reliability; "
                        "no discount (λ=1.0)",
                        group_keys, m,
                    )

        # Log summary for this group.
        by_source: dict[str, list[str]] = {}
        for m, src in sources.items():
            by_source.setdefault(src, []).append(m)
        logger.info("group=%s reliability sources: %s", group_keys, by_source)

        # Discount and normalise.
        discounted = {m: aw_w.get(m, 0.0) * lambda_k[m] for m in metric_set}
        total = sum(discounted.values())
        if total <= 0.0:
            n = len(metric_set)
            norm_w: dict[str, float] = {m: (1.0 / n if n > 0 else 0.0) for m in metric_set}
        else:
            norm_w = {m: v / total for m, v in discounted.items()}

        all_weights[group_keys] = norm_w
        all_sources[group_keys] = sources

    if return_sources:
        return all_weights, all_sources
    return all_weights
