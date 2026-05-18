"""Tests for src/metrics/redundancy.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.redundancy import (
    METRIC_CATEGORIES,
    category_redundancy_summary,
    compute_redundancy_matrix,
    prune_redundant_metrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_df(
    metrics: list[str],
    scores: dict[str, list[float]],
    n_images: int = 12,
    fae_methods: list[str] | None = None,
    model: str = "resnet18",
) -> pd.DataFrame:
    """Build a minimal long-format DataFrame for testing."""
    if fae_methods is None:
        fae_methods = ["saliency", "gradcam", "integrated_gradients"]

    rows = []
    for img_idx in range(n_images):
        img_id = f"ISIC_{img_idx:07d}"
        for fae in fae_methods:
            for metric in metrics:
                val_list = scores[metric]
                val = val_list[(img_idx * len(fae_methods) + fae_methods.index(fae)) % len(val_list)]
                rows.append(
                    {
                        "model": model,
                        "image_id": img_id,
                        "fae_method": fae,
                        "metric": metric,
                        "score": val,
                    }
                )
    return pd.DataFrame(rows)


# ── compute_redundancy_matrix ──────────────────────────────────────────────

class TestComputeRedundancyMatrix:
    def test_returns_one_matrix_per_model(self):
        """Two models produce two separate correlation matrices."""
        df = pd.concat(
            [
                _make_df(
                    ["faithfulness_correlation", "max_sensitivity"],
                    {"faithfulness_correlation": [0.1, 0.5, 0.9], "max_sensitivity": [0.9, 0.5, 0.1]},
                    model="resnet18",
                ),
                _make_df(
                    ["faithfulness_correlation", "max_sensitivity"],
                    {"faithfulness_correlation": [0.2, 0.4, 0.8], "max_sensitivity": [0.8, 0.4, 0.2]},
                    model="squeezenet",
                ),
            ]
        )
        result = compute_redundancy_matrix(df, group_by=("model",))
        assert len(result) == 2
        assert ("resnet18",) in result
        assert ("squeezenet",) in result

    def test_matrix_is_square_and_symmetric(self):
        """Returned matrix is square, symmetric, and has ones on diagonal."""
        df = _make_df(
            ["faithfulness_correlation", "max_sensitivity", "sparseness"],
            {
                "faithfulness_correlation": list(np.random.default_rng(0).random(36)),
                "max_sensitivity": list(np.random.default_rng(1).random(36)),
                "sparseness": list(np.random.default_rng(2).random(36)),
            },
        )
        result = compute_redundancy_matrix(df)
        mat = result[("resnet18",)]
        assert mat.shape[0] == mat.shape[1]
        assert mat.shape[0] == 3
        np.testing.assert_allclose(mat.values, mat.values.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(mat.values), np.ones(3), atol=1e-12)

    def test_min_obs_drops_all_nan_metric(self):
        """A metric with all-NaN scores is excluded when min_obs > 0."""
        base_df = _make_df(
            ["faithfulness_correlation", "max_sensitivity"],
            {"faithfulness_correlation": [0.3, 0.6, 0.9], "max_sensitivity": [0.1, 0.5, 0.9]},
        )
        # Add an all-NaN metric
        nan_rows = base_df[base_df["metric"] == "faithfulness_correlation"].copy()
        nan_rows["metric"] = "non_sensitivity"
        nan_rows["score"] = float("nan")
        df = pd.concat([base_df, nan_rows], ignore_index=True)

        result = compute_redundancy_matrix(df, min_obs=30)
        mat = result[("resnet18",)]
        assert "non_sensitivity" not in mat.columns

    def test_invalid_method_raises(self):
        df = _make_df(
            ["faithfulness_correlation", "max_sensitivity"],
            {"faithfulness_correlation": [0.5], "max_sensitivity": [0.5]},
        )
        with pytest.raises(ValueError, match="method must be"):
            compute_redundancy_matrix(df, method="kendall")

    def test_known_correlation_value(self):
        """Perfectly positively correlated metrics should yield ρ = 1."""
        rng = np.random.default_rng(42)
        base = rng.random(36).tolist()
        df = _make_df(
            ["faithfulness_correlation", "pixel_flipping"],
            {"faithfulness_correlation": base, "pixel_flipping": base},
        )
        result = compute_redundancy_matrix(df)
        mat = result[("resnet18",)]
        rho = mat.at["faithfulness_correlation", "pixel_flipping"]
        assert abs(rho - 1.0) < 1e-6


# ── prune_redundant_metrics ────────────────────────────────────────────────

class TestPruneRedundantMetrics:
    def _make_corr(self, metrics: list[str], rho_12: float) -> pd.DataFrame:
        """Identity matrix with one off-diagonal pair set to rho_12."""
        n = len(metrics)
        arr = np.eye(n)
        arr[0, 1] = arr[1, 0] = rho_12
        return pd.DataFrame(arr, index=metrics, columns=metrics)

    def test_no_redundant_pair_survives_intact(self):
        """When |ρ| < threshold, all metrics survive."""
        metrics = ["faithfulness_correlation", "max_sensitivity", "sparseness"]
        corr = self._make_corr(metrics, rho_12=0.60)
        kept, pruned = prune_redundant_metrics(corr, threshold=0.85)
        assert sorted(kept) == sorted(metrics)
        assert pruned == []

    def test_redundant_pair_drops_one_metric(self):
        """When |ρ| > threshold, exactly one of a redundant pair is dropped."""
        metrics = ["faithfulness_correlation", "pixel_flipping", "sparseness"]
        corr = self._make_corr(metrics, rho_12=0.92)
        kept, pruned = prune_redundant_metrics(corr, threshold=0.85)
        assert len(kept) == 2
        assert len(pruned) == 1
        dropped, retained, rho = pruned[0]
        assert dropped in metrics
        assert retained in metrics
        assert dropped != retained
        assert abs(rho - 0.92) < 1e-9

    def test_empty_corr_returns_empty(self):
        kept, pruned = prune_redundant_metrics(pd.DataFrame(), threshold=0.85)
        assert kept == []
        assert pruned == []

    def test_pruning_order_preserved(self):
        """Surviving metrics appear in the same order as the original columns."""
        metrics = ["a", "b", "c", "d"]
        arr = np.eye(4)
        arr[0, 1] = arr[1, 0] = 0.95  # a–b redundant
        corr = pd.DataFrame(arr, index=metrics, columns=metrics)
        kept, _ = prune_redundant_metrics(corr, threshold=0.85)
        # Kept order must be a subsequence of original order
        remaining_in_order = [m for m in metrics if m in kept]
        assert kept == remaining_in_order

    def test_greedy_drops_more_correlated_metric(self):
        """The metric more correlated to *all others* is dropped first."""
        # c is highly correlated with both a and b; a and b are uncorrelated.
        # When a–c pair triggers, mean|ρ| for c > mean|ρ| for a → c is dropped.
        metrics = ["a", "b", "c"]
        arr = np.array(
            [
                [1.0, 0.1, 0.92],  # a
                [0.1, 1.0, 0.93],  # b
                [0.92, 0.93, 1.0],  # c
            ]
        )
        corr = pd.DataFrame(arr, index=metrics, columns=metrics)
        kept, pruned = prune_redundant_metrics(corr, threshold=0.85)
        assert "c" not in kept
        assert "a" in kept
        assert "b" in kept


# ── category_redundancy_summary ────────────────────────────────────────────

class TestCategoryRedundancySummary:
    def _two_metric_corr(self, rho: float, metrics: list[str]) -> pd.DataFrame:
        arr = np.array([[1.0, rho], [rho, 1.0]])
        return pd.DataFrame(arr, index=metrics, columns=metrics)

    def test_within_category_rho_computed(self):
        """Single pair in Faithfulness: mean_abs_rho = |rho|."""
        corr = self._two_metric_corr(
            0.72, ["faithfulness_correlation", "pixel_flipping"]
        )
        summary = category_redundancy_summary(corr, METRIC_CATEGORIES)
        assert "Faithfulness" in summary.index
        assert abs(summary.at["Faithfulness", "mean_abs_rho"] - 0.72) < 1e-9

    def test_single_metric_category_yields_nan(self):
        """A category with only one metric in the matrix gets NaN mean_abs_rho."""
        corr = self._two_metric_corr(
            0.5, ["faithfulness_correlation", "sparseness"]
        )
        summary = category_redundancy_summary(corr, METRIC_CATEGORIES)
        # Complexity has only one metric (sparseness) in the matrix
        assert "Complexity" in summary.index
        assert np.isnan(summary.at["Complexity", "mean_abs_rho"])

    def test_returns_correct_n_metrics(self):
        """n_metrics reflects how many metrics from each category are in the matrix."""
        metrics = [
            "faithfulness_correlation",
            "pixel_flipping",
            "max_sensitivity",
            "avg_sensitivity",
        ]
        arr = np.eye(4)
        corr = pd.DataFrame(arr, index=metrics, columns=metrics)
        summary = category_redundancy_summary(corr, METRIC_CATEGORIES)
        assert summary.at["Faithfulness", "n_metrics"] == 2
        assert summary.at["Robustness", "n_metrics"] == 2

    def test_empty_corr_returns_empty_df(self):
        summary = category_redundancy_summary(pd.DataFrame(), METRIC_CATEGORIES)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
