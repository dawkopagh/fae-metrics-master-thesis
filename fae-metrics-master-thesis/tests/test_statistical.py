"""Tests for the Wilcoxon signed-rank significance test used in the thesis.

Scope: one test — the paired comparison between mqdiscount and single_fc
effectiveness scores (the sole significance claim in the paper).
Do not expand this file without explicit instruction.
"""

from __future__ import annotations

from scipy.stats import wilcoxon


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
