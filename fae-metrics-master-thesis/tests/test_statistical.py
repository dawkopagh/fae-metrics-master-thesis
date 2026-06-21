"""Tests for the Wilcoxon signed-rank significance test used in the thesis.

Scope: one test — the paired comparison between mqdiscount and single_fc
effectiveness scores (the sole significance claim in the paper).
Do not expand this file without explicit instruction.

This file was extended (per explicit thesis-task instruction) to cover the
full src/comparison/statistical_tests.py module: Friedman, Nemenyi post-hoc,
the wilcoxon_paired wrapper, and the CD-summary helper. Fixtures are small
and deterministic (fixed seeds only where randomness is intentional).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from src.comparison.statistical_tests import (
    build_cd_summary,
    friedman_test,
    nemenyi_posthoc,
    wilcoxon_paired,
)


def test_wilcoxon_monotonic_data_significant():
    """Wilcoxon signed-rank on monotonically ordered paired data → p < 0.05.

    Synthetic data: x > y for every pair by a constant positive offset,
    which is the strongest possible signal for the one-sided test.
    Verifies that the test statistic and p-value are computed correctly
    and that the scipy API behaves as expected for this use case.
    """
    # x = mqdiscount proxy: 15 values each 10 units above the paired y
    x = [float(v) for v in range(10, 25)]   # [10, 11, ..., 24]
    y = [float(v) for v in range(0, 15)]    # [ 0,  1, ..., 14]

    stat, p = wilcoxon(x, y)

    assert p < 0.05, (
        f"Expected p < 0.05 for monotonically ordered paired data, got p={p:.4f}"
    )
    assert stat >= 0.0, f"Wilcoxon statistic must be non-negative, got {stat}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _long_df(method_means: dict[str, float], n_images: int, noise: float = 0.0,
             seed: int = 0) -> pd.DataFrame:
    """Build a long-format effectiveness DataFrame.

    One model, *n_images* images, one row per (image, method). Each method's
    score is its mean plus optional Gaussian noise (fixed seed).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for img in range(n_images):
        for method, mean in method_means.items():
            score = mean + (rng.normal(0.0, noise) if noise > 0 else 0.0)
            rows.append(
                {
                    "model": "resnet18",
                    "image_id": f"img_{img:03d}",
                    "fae_method": method,
                    "effectiveness_index": score,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def test_friedman_identical_distributions_not_significant():
    """Identical per-image score profiles across methods → not significant."""
    # Every method gets the SAME score on a given image (image-varying but
    # method-invariant), so there is no systematic rank difference.
    rows = []
    for img in range(20):
        base = float(img)
        for method in ["a", "b", "c", "d"]:
            rows.append(
                {
                    "model": "m",
                    "image_id": f"i{img}",
                    "fae_method": method,
                    "effectiveness_index": base,
                }
            )
    df = pd.DataFrame(rows)
    res = friedman_test(df, block_cols=("model", "image_id"))
    assert res["p_value"] > 0.05
    assert res["n_methods"] == 4
    assert res["n_blocks"] == 20


def test_friedman_clearly_separated_significant():
    """Cleanly separated method means → significant Friedman result."""
    df = _long_df({"a": 0.1, "b": 0.5, "c": 0.9, "d": 1.3}, n_images=30,
                  noise=0.01, seed=42)
    res = friedman_test(df, block_cols=("model", "image_id"))
    assert res["p_value"] < 0.05
    # Best (rank 1) should be the highest-mean method when higher_is_better.
    rt = res["rank_table"]
    assert rt.loc["d", "rank_position"] == 1
    assert rt.loc["a", "rank_position"] == 4
    assert rt.loc["d", "mean_rank"] < rt.loc["a", "mean_rank"]


def test_friedman_higher_is_better_false_flips_ranking():
    """higher_is_better=False makes the lowest-mean method rank 1."""
    df = _long_df({"a": 0.1, "b": 0.5, "c": 0.9}, n_images=10, noise=0.01)
    res = friedman_test(df, higher_is_better=False)
    assert res["rank_table"].loc["a", "rank_position"] == 1


def test_friedman_drops_incomplete_blocks():
    """Blocks with a NaN for any method are dropped listwise and counted."""
    df = _long_df({"a": 0.2, "b": 0.5, "c": 0.8}, n_images=10, noise=0.01, seed=1)
    # Corrupt one image: drop method 'b' so that block is incomplete.
    df = df[~((df["image_id"] == "img_005") & (df["fae_method"] == "b"))]
    res = friedman_test(df)
    assert res["n_blocks"] == 9
    assert res["n_blocks_dropped"] == 1


def test_friedman_too_few_methods_raises():
    """Friedman is undefined for fewer than 3 methods → ValueError."""
    df = _long_df({"a": 0.1, "b": 0.9}, n_images=5)
    with pytest.raises(ValueError):
        friedman_test(df)


# ---------------------------------------------------------------------------
# Nemenyi post-hoc
# ---------------------------------------------------------------------------

def test_nemenyi_separates_extremes():
    """Nemenyi flags the widely-separated method pair as significant."""
    df = _long_df({"a": 0.1, "b": 0.5, "c": 0.9, "d": 1.3}, n_images=40,
                  noise=0.01, seed=7)
    res = nemenyi_posthoc(df)
    pm = res["p_values"]
    sig = res["significant"]
    # a vs d are the extremes — must be significant with a small p-value.
    assert sig.at["a", "d"]
    assert pm.at["a", "d"] < 0.05
    # Matrix symmetry and diagonal conventions.
    assert pm.at["a", "d"] == pm.at["d", "a"]
    assert pm.at["a", "a"] == pytest.approx(1.0)
    assert res["critical_difference"] > 0.0


def test_nemenyi_identical_not_significant():
    """No method differences → no significant pairs, CD not exceeded."""
    rows = []
    for img in range(20):
        for method in ["a", "b", "c"]:
            rows.append(
                {
                    "model": "m",
                    "image_id": f"i{img}",
                    "fae_method": method,
                    "effectiveness_index": float(img),
                }
            )
    df = pd.DataFrame(rows)
    res = nemenyi_posthoc(df)
    assert not res["significant"].to_numpy().any()


def test_nemenyi_critical_difference_formula():
    """CD matches q * sqrt(k(k+1)/(6N)) with the tabulated q at k, alpha."""
    df = _long_df({"a": 0.1, "b": 0.5, "c": 0.9}, n_images=25, noise=0.01)
    res = nemenyi_posthoc(df, alpha=0.05)
    k, n = res["n_methods"], res["n_blocks"]
    q = 2.343  # tabulated q_{0.05} for k=3
    expected = q * np.sqrt(k * (k + 1) / (6.0 * n))
    assert res["critical_difference"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# wilcoxon_paired wrapper
# ---------------------------------------------------------------------------

def test_wilcoxon_paired_positive_shift():
    """a systematically above b → significant, positive effect, direction a>b."""
    a = [float(v) for v in range(10, 25)]
    b = [float(v) for v in range(0, 15)]
    res = wilcoxon_paired(a, b)
    assert res["p_value"] < 0.05
    assert res["direction"] == "a>b"
    assert res["effect_size"] > 0.0
    assert res["n"] == 15


def test_wilcoxon_paired_negative_shift_direction():
    """b above a → direction a<b and negative effect size."""
    a = [float(v) for v in range(0, 15)]
    b = [float(v) for v in range(10, 25)]
    res = wilcoxon_paired(a, b)
    assert res["direction"] == "a<b"
    assert res["effect_size"] < 0.0


def test_wilcoxon_paired_matches_scipy_statistic():
    """Wrapper's statistic/p-value equal a direct scipy.wilcoxon call."""
    a = [3.0, 5.0, 2.0, 8.0, 7.0, 6.0, 9.0, 4.0]
    b = [1.0, 4.0, 3.0, 5.0, 6.0, 2.0, 7.0, 5.0]
    stat, p = wilcoxon(a, b)
    res = wilcoxon_paired(a, b)
    assert res["statistic"] == pytest.approx(stat)
    assert res["p_value"] == pytest.approx(p)


def test_wilcoxon_paired_drops_nan_pairs():
    """Pairs with NaN on either side are dropped (pairwise-complete)."""
    a = [10.0, 11.0, np.nan, 13.0, 14.0]
    b = [0.0, 1.0, 2.0, np.nan, 4.0]
    res = wilcoxon_paired(a, b)
    assert res["n"] == 3          # indices 0, 1, 4 survive
    assert res["n_dropped"] == 2
    assert res["direction"] == "a>b"


def test_wilcoxon_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        wilcoxon_paired([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# CD summary helper
# ---------------------------------------------------------------------------

def test_build_cd_summary_columns_and_order():
    """build_cd_summary yields one row per method, best-first, with context."""
    df = _long_df({"a": 0.1, "b": 0.5, "c": 0.9}, n_images=20, noise=0.01)
    fres = friedman_test(df)
    nres = nemenyi_posthoc(df)
    summary = build_cd_summary(fres, nres)
    assert list(summary["method"])[0] == "c"  # best mean rank first
    for col in [
        "method", "mean_rank", "rank_position", "critical_difference",
        "friedman_statistic", "friedman_p_value", "n_blocks", "n_methods",
    ]:
        assert col in summary.columns
    assert summary["critical_difference"].iloc[0] == pytest.approx(
        nres["critical_difference"]
    )
    # Without a Nemenyi result the CD is NaN but the table still builds.
    summary_no_cd = build_cd_summary(fres)
    assert np.isnan(summary_no_cd["critical_difference"].iloc[0])
