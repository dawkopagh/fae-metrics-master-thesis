"""
Statistical significance tests for FAE method comparison — Research Contributions 3 & 4.

Responsibilities (see docs/thesis_plan.md §2, Research Contributions 3–4;
§6, Statistical Comparison; Decision D5):
    - Friedman test + Nemenyi post-hoc for ranking FAE methods across metrics
      on the full test set (non-parametric, multi-method comparison)
    - Wilcoxon signed-rank test for paired comparison of individual vs.
      NormEnsembleXAI-ensembled attributions per metric
    - Return corrected p-values, critical difference diagrams, and rank tables
      formatted for Chapter 4 figures

Not responsible for: computing effectiveness indices (see aggregation/),
metric score computation (see metrics/),
or visualisation beyond raw figure export (thesis LaTeX figures are authored
separately in Latex/).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Studentised-range critical values q_{α,∞,k} (df = ∞, k treatments) used to
# form the Nemenyi critical difference. Values are the standard table entries
# divided by sqrt(2) (the form required by the CD formula). Keyed by k (number
# of methods compared). Source: Demšar (2006), Table 5; only the rows needed
# for the thesis (2–12 FAE methods) are tabulated.
_NEMENYI_Q_ALPHA: dict[float, dict[int, float]] = {
    0.05: {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
    },
    0.10: {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
        8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030,
    },
}


def _pivot_blocks(
    scores_df: pd.DataFrame,
    method_col: str,
    block_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Pivot long-format scores into a (block × method) wide matrix.

    Each row is one *block* (a matched observation, e.g. an image, or an
    image × metric pair) and each column is one FAE *method*. The Friedman
    and Nemenyi tests treat the rows as related samples (blocks) over which
    the methods are ranked.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Long-format scores.
    method_col : str
        Column whose unique values become the compared treatments (columns).
    block_cols : list[str]
        Columns whose combination identifies a matched block (the index).
    value_col : str
        Column holding the numeric score.

    Returns
    -------
    pd.DataFrame
        Wide matrix indexed by *block_cols*, one column per method. Cells
        with no observation (or duplicates collapsed by ``mean``) may be NaN.
    """
    wide = scores_df.pivot_table(
        index=block_cols,
        columns=method_col,
        values=value_col,
        aggfunc="mean",
    )
    wide.columns.name = None
    return wide


def _mean_rank_table(wide: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    """Compute the per-method mean rank over complete blocks.

    Ranks are assigned within each block (row); ties receive the average
    rank. ``ascending=False`` means a *higher* score earns rank 1 (best),
    which is the convention for effectiveness indices where larger is better.

    Returns a table indexed by method with columns ``mean_rank`` and
    ``rank_position`` (1 = best mean rank).
    """
    ranks = wide.rank(axis=1, ascending=ascending, method="average")
    mean_rank = ranks.mean(axis=0)
    table = mean_rank.to_frame("mean_rank").sort_values("mean_rank")
    table["rank_position"] = np.arange(1, len(table) + 1)
    return table


def friedman_test(
    scores_df: pd.DataFrame,
    method_col: str = "fae_method",
    block_cols: tuple[str, ...] = ("model", "image_id"),
    value_col: str = "effectiveness_index",
    higher_is_better: bool = True,
) -> dict:
    """Friedman test ranking FAE methods across matched blocks.

    The Friedman test is the non-parametric analogue of a repeated-measures
    ANOVA: it tests the null hypothesis that all FAE methods have the same
    distribution of ranks across the matched blocks (per Decision D5).
    Blocks default to ``(model, image_id)`` so every image contributes one
    matched comparison of all methods.

    Only blocks that are **complete** (a finite score for every method) enter
    the test, because the Friedman statistic requires a fully-crossed design.
    Incomplete blocks (any NaN) are dropped listwise and counted in the
    returned ``n_blocks_dropped``.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Long-format scores with at least *method_col*, *value_col*, and every
        column in *block_cols*.
    method_col : str
        Column whose unique values are the compared FAE methods.
    block_cols : tuple of str
        Columns whose combination identifies a matched block.
    value_col : str
        Column holding the numeric effectiveness score.
    higher_is_better : bool
        If ``True`` (default), a larger score is better and earns rank 1 in
        the mean-rank table. Does not affect the Friedman statistic/p-value
        (which are invariant to monotone rank direction) — only the reported
        ``rank_position`` ordering.

    Returns
    -------
    dict
        Keys:

        - ``statistic`` (float): Friedman chi-square statistic.
        - ``p_value`` (float): two-sided p-value.
        - ``n_methods`` (int): number of methods compared.
        - ``n_blocks`` (int): number of complete blocks used.
        - ``n_blocks_dropped`` (int): incomplete blocks discarded.
        - ``rank_table`` (pd.DataFrame): per-method ``mean_rank`` and
          ``rank_position`` (1 = best), indexed by method name.
        - ``methods`` (list[str]): method names in the column order tested.

    Raises
    ------
    ValueError
        If fewer than 3 methods (the Friedman test is undefined for two
        treatments — use :func:`wilcoxon_paired` instead) or fewer than 2
        complete blocks remain after dropping incomplete blocks.
    """
    wide = _pivot_blocks(scores_df, method_col, list(block_cols), value_col)

    n_total = len(wide)
    complete = wide.dropna(axis=0, how="any")
    n_blocks = len(complete)
    n_dropped = n_total - n_blocks
    methods = list(complete.columns)

    if len(methods) < 3:
        raise ValueError(
            f"Friedman test needs >= 3 methods, got {len(methods)}: {methods}. "
            "For two methods use wilcoxon_paired()."
        )
    if n_blocks < 2:
        raise ValueError(
            f"Friedman test needs >= 2 complete blocks, got {n_blocks} "
            f"({n_dropped} of {n_total} blocks dropped for incompleteness)"
        )

    if n_dropped:
        logger.info(
            "friedman_test: dropped %d of %d blocks with missing scores; "
            "%d complete blocks over %d methods used",
            n_dropped, n_total, n_blocks, len(methods),
        )

    columns = [complete[m].to_numpy(dtype=float) for m in methods]
    with np.errstate(invalid="ignore", divide="ignore"):
        statistic, p_value = stats.friedmanchisquare(*columns)

    # Degenerate fully-tied design: every method scores identically within
    # every block, so there is no rank variation and the statistic is 0/0.
    # Report a zero statistic and p = 1.0 (no evidence of any difference).
    if np.isnan(statistic) or np.isnan(p_value):
        logger.info(
            "friedman_test: degenerate (fully tied) design — reporting "
            "statistic=0.0, p=1.0"
        )
        statistic, p_value = 0.0, 1.0

    rank_table = _mean_rank_table(complete, ascending=not higher_is_better)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_methods": len(methods),
        "n_blocks": n_blocks,
        "n_blocks_dropped": n_dropped,
        "rank_table": rank_table,
        "methods": methods,
    }


def nemenyi_posthoc(
    scores_df: pd.DataFrame,
    method_col: str = "fae_method",
    block_cols: tuple[str, ...] = ("model", "image_id"),
    value_col: str = "effectiveness_index",
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> dict:
    """Nemenyi post-hoc test for all pairwise FAE-method comparisons.

    Run after a significant :func:`friedman_test`. The Nemenyi test compares
    every pair of methods using their mean ranks; two methods differ
    significantly if the absolute difference of their mean ranks exceeds the
    **critical difference** (CD):

    .. math::
        \\mathrm{CD} = q_{\\alpha,k}\\,\\sqrt{\\frac{k(k+1)}{6 N}}

    where *k* is the number of methods, *N* the number of complete blocks,
    and :math:`q_{\\alpha,k}` the studentised-range critical value (df = ∞)
    divided by :math:`\\sqrt 2` (tabulated in :data:`_NEMENYI_Q_ALPHA`).

    Pairwise p-values are obtained from the studentised-range distribution
    (``scipy.stats.studentized_range``) applied to the standardised
    mean-rank difference, matching the ``scikit_posthocs`` implementation.
    No further multiplicity correction is applied: the studentised-range
    critical value already controls the family-wise error rate across all
    pairwise comparisons.

    NaN handling is identical to :func:`friedman_test`: incomplete blocks
    (any method missing) are dropped listwise so all mean ranks are computed
    over the same *N* blocks.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Long-format scores (same schema as :func:`friedman_test`).
    method_col, block_cols, value_col : see :func:`friedman_test`.
    alpha : float
        Family-wise significance level. ``0.05`` or ``0.10`` use tabulated
        critical values; other values fall back to the exact studentised-range
        quantile from scipy.
    higher_is_better : bool
        Direction passed to the mean-rank table (does not affect p-values).

    Returns
    -------
    dict
        Keys:

        - ``p_values`` (pd.DataFrame): symmetric method × method matrix of
          two-sided p-values (diagonal = 1.0).
        - ``critical_difference`` (float): the CD threshold.
        - ``significant`` (pd.DataFrame): boolean method × method matrix,
          ``True`` where ``|R_i - R_j| > CD``.
        - ``rank_table`` (pd.DataFrame): per-method mean ranks (as in
          :func:`friedman_test`).
        - ``n_methods`` (int), ``n_blocks`` (int), ``alpha`` (float).

    Raises
    ------
    ValueError
        If fewer than 2 methods or fewer than 2 complete blocks remain.
    """
    wide = _pivot_blocks(scores_df, method_col, list(block_cols), value_col)
    complete = wide.dropna(axis=0, how="any")
    n_blocks = len(complete)
    methods = list(complete.columns)
    k = len(methods)

    if k < 2:
        raise ValueError(f"Nemenyi needs >= 2 methods, got {k}: {methods}")
    if n_blocks < 2:
        raise ValueError(f"Nemenyi needs >= 2 complete blocks, got {n_blocks}")

    # Mean ranks over the same complete blocks (consistent N for every pair).
    ranks = complete.rank(axis=1, ascending=not higher_is_better, method="average")
    mean_ranks = ranks.mean(axis=0)

    # Critical difference. q is the studentised-range value / sqrt(2).
    q = _critical_value(alpha, k)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    # Standard error of a mean-rank difference for the studentised range.
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    p_matrix = pd.DataFrame(
        np.ones((k, k)), index=methods, columns=methods, dtype=float
    )
    sig_matrix = pd.DataFrame(
        np.zeros((k, k), dtype=bool), index=methods, columns=methods
    )

    for i in range(k):
        for j in range(i + 1, k):
            mi, mj = methods[i], methods[j]
            diff = abs(mean_ranks[mi] - mean_ranks[mj])
            # Studentised-range statistic = diff / (se / sqrt(2)).
            q_stat = diff / (se / np.sqrt(2.0))
            p = float(
                stats.studentized_range.sf(q_stat, k, np.inf)
            )
            p = min(max(p, 0.0), 1.0)
            p_matrix.at[mi, mj] = p
            p_matrix.at[mj, mi] = p
            is_sig = diff > cd
            sig_matrix.at[mi, mj] = is_sig
            sig_matrix.at[mj, mi] = is_sig

    rank_table = _mean_rank_table(complete, ascending=not higher_is_better)

    return {
        "p_values": p_matrix,
        "critical_difference": float(cd),
        "significant": sig_matrix,
        "rank_table": rank_table,
        "n_methods": k,
        "n_blocks": n_blocks,
        "alpha": alpha,
    }


def _critical_value(alpha: float, k: int) -> float:
    """Resolve the Nemenyi q-value (studentised range / sqrt(2)).

    Uses the tabulated value in :data:`_NEMENYI_Q_ALPHA` when *alpha* and *k*
    are present, otherwise computes the exact studentised-range quantile with
    scipy (``ppf`` of ``studentized_range`` at df = ∞, divided by sqrt(2)).
    """
    table = _NEMENYI_Q_ALPHA.get(alpha)
    if table is not None and k in table:
        return table[k]
    # Exact fallback for non-tabulated alpha or k > 12.
    q_sr = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf))
    return q_sr / np.sqrt(2.0)


def wilcoxon_paired(
    a,
    b,
    alternative: str = "two-sided",
    drop_nan: bool = True,
    zero_method: str = "wilcox",
) -> dict:
    """Wilcoxon signed-rank test for a paired comparison.

    The thesis uses this for the paired comparison of two aggregation schemes
    (e.g. ``mqdiscount`` vs ``single_fc``) or individual vs. ensembled
    attributions per metric (Decision D5). It tests whether the median of the
    paired differences ``a - b`` is zero.

    Effect size is reported as the matched-pairs rank-biserial correlation
    ``r = (W+ - W-) / (W+ + W-)`` in ``[-1, 1]``; its sign matches the
    direction of the typical difference (``a > b`` ⇒ positive). ``direction``
    is one of ``'a>b'``, ``'a<b'``, or ``'tie'`` based on the sign of the sum
    of signed ranks.

    NaN handling: by default (``drop_nan=True``) any pair where either side is
    NaN is dropped (pairwise-complete), matching the notebook's
    ``dropna()`` before ``scipy.stats.wilcoxon``. Pairs whose difference is
    exactly zero are handled by scipy per *zero_method*.

    Parameters
    ----------
    a, b : array-like
        Paired observations of equal length. Converted to float arrays.
    alternative : {'two-sided', 'greater', 'less'}
        Passed to :func:`scipy.stats.wilcoxon`. ``'greater'`` tests
        ``median(a - b) > 0``.
    drop_nan : bool
        If ``True``, drop pairs where either value is NaN before testing.
    zero_method : {'wilcox', 'pratt', 'zsplit'}
        Forwarded to :func:`scipy.stats.wilcoxon` for handling zero
        differences. Default ``'wilcox'`` (scipy default: discard zeros).

    Returns
    -------
    dict
        Keys:

        - ``statistic`` (float): Wilcoxon W (smaller of the signed-rank sums).
        - ``p_value`` (float).
        - ``effect_size`` (float): rank-biserial correlation in ``[-1, 1]``.
        - ``direction`` (str): ``'a>b'``, ``'a<b'``, or ``'tie'``.
        - ``n`` (int): number of pairs used.
        - ``n_dropped`` (int): pairs dropped for NaN (0 if ``drop_nan`` False).
        - ``n_zero`` (int): pairs with zero difference.
        - ``alternative`` (str): the *alternative* used.

    Raises
    ------
    ValueError
        If *a* and *b* differ in length, or fewer than 1 valid pair remains.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"a and b must have the same shape, got {arr_a.shape} vs {arr_b.shape}"
        )

    n_input = arr_a.size
    if drop_nan:
        mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
        arr_a, arr_b = arr_a[mask], arr_b[mask]
    n_dropped = n_input - arr_a.size

    if arr_a.size < 1:
        raise ValueError("No valid paired observations after NaN removal")

    diff = arr_a - arr_b
    n_zero = int(np.sum(diff == 0.0))

    if n_dropped:
        logger.info(
            "wilcoxon_paired: dropped %d of %d pairs containing NaN",
            n_dropped, n_input,
        )

    statistic, p_value = stats.wilcoxon(
        arr_a, arr_b, alternative=alternative, zero_method=zero_method
    )

    # Rank-biserial effect size from signed ranks of the (nonzero) differences.
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        effect_size = 0.0
        direction = "tie"
    else:
        ranks = stats.rankdata(np.abs(nonzero))
        w_plus = float(ranks[nonzero > 0].sum())
        w_minus = float(ranks[nonzero < 0].sum())
        total = w_plus + w_minus
        effect_size = (w_plus - w_minus) / total if total > 0 else 0.0
        if w_plus > w_minus:
            direction = "a>b"
        elif w_plus < w_minus:
            direction = "a<b"
        else:
            direction = "tie"

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "direction": direction,
        "n": int(arr_a.size),
        "n_dropped": int(n_dropped),
        "n_zero": n_zero,
        "alternative": alternative,
    }


def build_cd_summary(
    friedman_result: dict,
    nemenyi_result: dict | None = None,
) -> pd.DataFrame:
    """Assemble a rank / critical-difference summary table for Chapter 4.

    Produces a data-first table (no plotting) carrying everything a LaTeX
    critical-difference diagram or rank table needs: per-method mean ranks,
    rank positions, and the shared CD / Friedman context as repeated columns
    so the CSV is self-describing. Visualisation is authored separately in
    ``Latex/`` per the module docstring.

    Parameters
    ----------
    friedman_result : dict
        Output of :func:`friedman_test`.
    nemenyi_result : dict or None
        Output of :func:`nemenyi_posthoc`. If provided, the critical
        difference is included; otherwise ``critical_difference`` is NaN.

    Returns
    -------
    pd.DataFrame
        One row per method, sorted by ascending mean rank (best first).
        Columns: ``method``, ``mean_rank``, ``rank_position``,
        ``critical_difference``, ``friedman_statistic``, ``friedman_p_value``,
        ``n_blocks``, ``n_methods``.
    """
    rank_table = friedman_result["rank_table"]
    cd = (
        nemenyi_result["critical_difference"]
        if nemenyi_result is not None
        else float("nan")
    )

    summary = rank_table.reset_index()
    summary = summary.rename(columns={summary.columns[0]: "method"})
    summary["critical_difference"] = cd
    summary["friedman_statistic"] = friedman_result["statistic"]
    summary["friedman_p_value"] = friedman_result["p_value"]
    summary["n_blocks"] = friedman_result["n_blocks"]
    summary["n_methods"] = friedman_result["n_methods"]
    return summary.sort_values("mean_rank").reset_index(drop=True)
