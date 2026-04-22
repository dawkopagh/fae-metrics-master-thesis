"""Tests for src/aggregation/aggregate.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aggregation.aggregate import (
    aggregate_scores,
    geometric_mean,
    minimum,
    weighted_mean,
)


# ── Scalar: weighted_mean ────────────────────────────────────────────────

class TestWeightedMean:
    def test_uniform_weights_equals_numpy_mean(self):
        s = np.array([0.2, 0.4, 0.6, 0.8])
        assert weighted_mean(s) == pytest.approx(np.mean(s))

    def test_explicit_uniform_weights(self):
        s = np.array([0.2, 0.4, 0.6, 0.8])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert weighted_mean(s, w) == pytest.approx(np.mean(s))

    def test_non_uniform_weights(self):
        """Hand calculation: 0.7 * 0.8 + 0.2 * 0.4 + 0.1 * 0.1 = 0.56 + 0.08 + 0.01 = 0.65."""
        s = np.array([0.8, 0.4, 0.1])
        w = np.array([0.7, 0.2, 0.1])
        assert weighted_mean(s, w) == pytest.approx(0.65)

    def test_weights_not_summing_to_one_raises(self):
        s = np.array([0.5, 0.5])
        w = np.array([0.3, 0.3])  # sum = 0.6
        with pytest.raises(ValueError, match="sum to 1.0"):
            weighted_mean(s, w)


# ── Scalar: minimum ──────────────────────────────────────────────────────

class TestMinimum:
    def test_returns_min(self):
        s = np.array([0.9, 0.1, 0.5])
        assert minimum(s) == pytest.approx(0.1)

    def test_weights_ignored(self):
        s = np.array([0.9, 0.1, 0.5])
        w = np.array([0.0, 0.0, 1.0])
        # Should still return 0.1 despite weight on 0.5
        assert minimum(s, w) == pytest.approx(0.1)


# ── Scalar: geometric_mean ───────────────────────────────────────────────

class TestGeometricMean:
    def test_all_ones(self):
        """geometric_mean([1, 1, 1]) ≈ 1."""
        s = np.array([1.0, 1.0, 1.0])
        assert geometric_mean(s) == pytest.approx(1.0, abs=1e-9)

    def test_hand_calculation(self):
        """[2, 8]: unweighted geometric mean = sqrt(2*8) = 4.
        With eps=1e-12 the answer should be very close.
        """
        s = np.array([2.0, 8.0])
        expected = np.sqrt(2.0 * 8.0)
        assert geometric_mean(s) == pytest.approx(expected, rel=1e-9)

    def test_negative_input_raises(self):
        s = np.array([0.5, -0.1, 0.3])
        with pytest.raises(ValueError, match="normalised scores >= 0"):
            geometric_mean(s)

    def test_with_weights(self):
        """[4, 16] with weights [0.75, 0.25].
        exp(0.75*ln(4) + 0.25*ln(16)) = exp(0.75*1.386 + 0.25*2.773)
        = exp(1.040 + 0.693) = exp(1.733) ≈ 5.657 = 4^0.75 * 16^0.25.
        """
        s = np.array([4.0, 16.0])
        w = np.array([0.75, 0.25])
        expected = 4.0 ** 0.75 * 16.0 ** 0.25
        assert geometric_mean(s, w) == pytest.approx(expected, rel=1e-9)


# ── DataFrame-level: aggregate_scores ────────────────────────────────────

class TestAggregateScores:
    @pytest.fixture()
    def fixture_df(self) -> pd.DataFrame:
        """Small fixture: 2 groups × 3 metrics each."""
        return pd.DataFrame({
            "model": ["m1"] * 6,
            "image_id": ["i1"] * 3 + ["i1"] * 3,
            "fae_method": ["f1"] * 3 + ["f2"] * 3,
            "metric": [
                "faithfulness_correlation", "max_sensitivity", "relevance_mass_accuracy",
                "faithfulness_correlation", "max_sensitivity", "relevance_mass_accuracy",
            ],
            "score_normalized": [0.8, 0.6, 0.4, 0.2, 0.9, 0.7],
        })

    def test_correct_number_of_rows(self, fixture_df: pd.DataFrame):
        result = aggregate_scores(fixture_df, operator="weighted_mean")
        # 2 groups: (m1, i1, f1), (m1, i1, f2)
        assert len(result) == 2

    def test_output_columns(self, fixture_df: pd.DataFrame):
        result = aggregate_scores(fixture_df, operator="weighted_mean")
        expected_cols = {"model", "image_id", "fae_method",
                         "effectiveness_index", "operator"}
        assert set(result.columns) == expected_cols

    def test_operator_column_value(self, fixture_df: pd.DataFrame):
        result = aggregate_scores(fixture_df, operator="min")
        assert (result["operator"] == "min").all()

    def test_weighted_mean_values(self, fixture_df: pd.DataFrame):
        result = aggregate_scores(fixture_df, operator="weighted_mean")
        f1_row = result[result["fae_method"] == "f1"]
        assert f1_row["effectiveness_index"].values[0] == pytest.approx(
            np.mean([0.8, 0.6, 0.4])
        )

    def test_missing_score_normalized_raises(self):
        df = pd.DataFrame({
            "model": ["m1"],
            "image_id": ["i1"],
            "fae_method": ["f1"],
            "metric": ["faithfulness_correlation"],
            "score": [0.5],  # NOT score_normalized
        })
        with pytest.raises(ValueError, match="score_normalized"):
            aggregate_scores(df)

    def test_with_metric_weights(self, fixture_df: pd.DataFrame):
        weights = {
            "faithfulness_correlation": 0.5,
            "max_sensitivity": 0.3,
            "relevance_mass_accuracy": 0.2,
        }
        result = aggregate_scores(
            fixture_df, operator="weighted_mean", weights=weights,
        )
        f1_row = result[result["fae_method"] == "f1"]
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4
        assert f1_row["effectiveness_index"].values[0] == pytest.approx(expected)

    def test_unknown_operator_raises(self, fixture_df: pd.DataFrame):
        with pytest.raises(ValueError, match="Unknown operator"):
            aggregate_scores(fixture_df, operator="median")
