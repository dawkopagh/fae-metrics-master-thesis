"""Tests for src/aggregation/normalize.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aggregation.normalize import (
    METRIC_DIRECTIONS,
    min_max,
    normalize_scores,
    second_moment_scale,
    z_score,
)


# ── Element-wise: second_moment_scale ────────────────────────────────────

class TestSecondMomentScale:
    def test_fixed_input_direction_positive(self):
        """Hand-computed: scores = [3, 4], rms = sqrt((9+16)/2) = sqrt(12.5).
        Expected: [3/sqrt(12.5), 4/sqrt(12.5)] * (+1).
        """
        scores = np.array([3.0, 4.0])
        rms = np.sqrt(12.5)
        expected = np.array([3.0 / rms, 4.0 / rms])
        result = second_moment_scale(scores, direction=+1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_fixed_input_direction_negative(self):
        scores = np.array([3.0, 4.0])
        rms = np.sqrt(12.5)
        expected = np.array([-3.0 / rms, -4.0 / rms])
        result = second_moment_scale(scores, direction=-1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_direction_flips_order(self):
        """Direction +1 vs. -1 produces opposite-ordered outputs."""
        scores = np.array([1.0, 2.0, 3.0])
        pos = second_moment_scale(scores, +1)
        neg = second_moment_scale(scores, -1)
        # +1 preserves order: pos[0] < pos[1] < pos[2]
        assert pos[0] < pos[1] < pos[2]
        # -1 reverses: neg[0] > neg[1] > neg[2]
        assert neg[0] > neg[1] > neg[2]

    def test_all_identical_returns_constant(self):
        """All-identical input: rms = |c|, so result = sign(c)*direction for nonzero c.
        For c = 5: rms = 5, result = [1, 1] * direction.
        """
        scores = np.array([5.0, 5.0, 5.0])
        result = second_moment_scale(scores, +1)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], rtol=1e-12)

    def test_all_zeros(self):
        scores = np.array([0.0, 0.0])
        result = second_moment_scale(scores, +1)
        np.testing.assert_array_equal(result, [0.0, 0.0])


# ── Element-wise: z_score ────────────────────────────────────────────────

class TestZScore:
    def test_fixed_input(self):
        """scores = [2, 4, 6]. mean=4, std=sqrt(8/3).
        z = [-2, 0, 2] / sqrt(8/3) * (+1).
        """
        scores = np.array([2.0, 4.0, 6.0])
        mu = 4.0
        sigma = np.std(scores, ddof=0)
        expected = (scores - mu) / sigma
        result = z_score(scores, +1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_direction_negative(self):
        scores = np.array([2.0, 4.0, 6.0])
        pos = z_score(scores, +1)
        neg = z_score(scores, -1)
        np.testing.assert_allclose(neg, -pos, rtol=1e-12)

    def test_direction_flips_order(self):
        scores = np.array([1.0, 3.0, 5.0])
        pos = z_score(scores, +1)
        neg = z_score(scores, -1)
        assert pos[0] < pos[2]
        assert neg[0] > neg[2]

    def test_all_identical_returns_zeros(self):
        """std == 0 → return zeros."""
        scores = np.array([7.0, 7.0, 7.0])
        result = z_score(scores, +1)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])


# ── Element-wise: min_max ────────────────────────────────────────────────

class TestMinMax:
    def test_fixed_input_direction_positive(self):
        """scores = [10, 20, 30]. min=10, max=30.
        Expected: [0, 0.5, 1].
        """
        scores = np.array([10.0, 20.0, 30.0])
        expected = np.array([0.0, 0.5, 1.0])
        result = min_max(scores, +1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_fixed_input_direction_negative(self):
        """direction=-1 flips: [1, 0.5, 0]."""
        scores = np.array([10.0, 20.0, 30.0])
        expected = np.array([1.0, 0.5, 0.0])
        result = min_max(scores, -1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_direction_flips_order(self):
        scores = np.array([1.0, 5.0, 9.0])
        pos = min_max(scores, +1)
        neg = min_max(scores, -1)
        assert pos[0] < pos[2]
        assert neg[0] > neg[2]

    def test_all_identical_returns_half(self):
        """max == min → return 0.5 for all."""
        scores = np.array([3.0, 3.0, 3.0])
        result_pos = min_max(scores, +1)
        result_neg = min_max(scores, -1)
        np.testing.assert_array_equal(result_pos, [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(result_neg, [0.5, 0.5, 0.5])

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(-10, 10, 100)
        for d in (+1, -1):
            normed = min_max(scores, d)
            assert normed.min() >= 0.0 - 1e-15
            assert normed.max() <= 1.0 + 1e-15


# ── Unknown metric KeyError ──────────────────────────────────────────────

class TestUnknownMetric:
    def test_unknown_metric_raises_keyerror(self):
        df = pd.DataFrame({
            "model": ["resnet18"],
            "image_id": ["img_0"],
            "fae_method": ["saliency"],
            "metric": ["bogus_metric"],
            "score": [0.5],
        })
        with pytest.raises(KeyError, match="bogus_metric"):
            normalize_scores(df, method="min_max")


# ── DataFrame-level normalize_scores ─────────────────────────────────────

class TestNormalizeScores:
    def test_groups_normalize_independently(self):
        """Group A range [0, 1], Group B range [100, 200].
        With min_max, both should map to [0, 1].
        """
        df = pd.DataFrame({
            "model": ["m1"] * 4,
            "image_id": ["i1", "i2", "i1", "i2"],
            "fae_method": ["f1", "f1", "f1", "f1"],
            "metric": [
                "faithfulness_correlation", "faithfulness_correlation",
                "relevance_mass_accuracy", "relevance_mass_accuracy",
            ],
            "score": [0.0, 1.0, 100.0, 200.0],
        })
        result = normalize_scores(df, method="min_max")
        assert "score_normalized" in result.columns

        grp_a = result[result["metric"] == "faithfulness_correlation"]["score_normalized"]
        grp_b = result[result["metric"] == "relevance_mass_accuracy"]["score_normalized"]

        # Both groups normalized to [0, 1] independently
        np.testing.assert_allclose(sorted(grp_a), [0.0, 1.0])
        np.testing.assert_allclose(sorted(grp_b), [0.0, 1.0])

    def test_original_score_preserved(self):
        df = pd.DataFrame({
            "model": ["m1", "m1"],
            "image_id": ["i1", "i2"],
            "fae_method": ["f1", "f1"],
            "metric": ["faithfulness_correlation", "faithfulness_correlation"],
            "score": [0.3, 0.7],
        })
        result = normalize_scores(df, method="z_score")
        pd.testing.assert_series_equal(result["score"], df["score"])

    def test_invalid_method_raises(self):
        df = pd.DataFrame({
            "model": ["m1"],
            "image_id": ["i1"],
            "fae_method": ["f1"],
            "metric": ["faithfulness_correlation"],
            "score": [0.5],
        })
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_scores(df, method="banana")

    def test_accepts_pandas_series_input(self):
        series = pd.Series([1.0, 2.0, 3.0])
        result = second_moment_scale(series, +1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
