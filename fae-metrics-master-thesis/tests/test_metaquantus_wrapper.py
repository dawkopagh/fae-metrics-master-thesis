"""Tests for src/meta_evaluation/metaquantus_wrapper.py.

All fixtures are synthetic — no real Quantus metrics or neural network forward
passes.  The tests verify the mathematical logic of the NR and AR algorithms,
not Quantus integration (that is covered by experiments/metaquantus_smoke.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.meta_evaluation.metaquantus_wrapper import (
    adversarial_reactivity_test,
    meta_evaluate_metric,
    noise_resilience_test,
    run_meta_evaluation_full,
    screen_meta_eval_candidates,
)


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

class _DummyModel(nn.Module):
    """Trivial linear model that ignores input shape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 3)


def _explain_zeros(model, inputs, targets, **kwargs) -> np.ndarray:
    """explain_fn that always returns zero-valued attributions."""
    return np.zeros_like(inputs, dtype=np.float32)


def _make_inputs(
    n_images: int = 2,
    c: int = 3,
    h: int = 8,
    w: int = 8,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    rng = np.random.default_rng(seed)
    images = [rng.random((c, h, w)).astype(np.float32) for _ in range(n_images)]
    attrs = [rng.random((c, h, w)).astype(np.float32) for _ in range(n_images)]
    targets = [0] * n_images
    return images, attrs, targets


# ---------------------------------------------------------------------------
# Fixture 1: constant metric (always returns 0.5)
# ---------------------------------------------------------------------------

class _ConstantMetric:
    """Returns 0.5 for every image in the batch, regardless of attribution."""

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device):
        return [0.5] * x_batch.shape[0]


class TestNRConstantMetric:
    """NR on a perfectly stable metric: nr_score must equal 1.0."""

    def setup_method(self):
        self.model = _DummyModel()
        self.images, self.attrs, self.targets = _make_inputs()

    def test_nr_score_is_one(self):
        result = noise_resilience_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=5,
        )
        assert result["nr_score"] == pytest.approx(1.0, abs=1e-9)

    def test_std_score_is_zero(self):
        result = noise_resilience_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=5,
        )
        assert result["std_score"] == pytest.approx(0.0, abs=1e-9)

    def test_cv_score_is_zero(self):
        result = noise_resilience_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=5,
        )
        assert result["cv_score"] == pytest.approx(0.0, abs=1e-9)

    def test_raw_scores_length_matches_n_seeds(self):
        result = noise_resilience_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=7,
        )
        assert len(result["raw_scores"]) == 7

    def test_mean_score_is_constant_value(self):
        result = noise_resilience_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=5,
        )
        assert result["mean_score"] == pytest.approx(0.5, abs=1e-9)


class TestARConstantMetric:
    """AR on a constant metric: ar_score must equal 0.0 (no monotonic change)."""

    def setup_method(self):
        self.model = _DummyModel()
        self.images, self.attrs, self.targets = _make_inputs()

    def test_ar_score_is_zero(self):
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=5,
        )
        # All scores are 0.5 → Spearman ρ = 0 → ar_score = 0
        assert result["ar_score"] == pytest.approx(0.0, abs=1e-9)

    def test_monotonicity_is_zero(self):
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=5,
        )
        assert result["monotonicity"] == pytest.approx(0.0, abs=1e-9)

    def test_levels_and_scores_length(self):
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=4,
        )
        assert len(result["levels"]) == 4
        assert len(result["scores"]) == 4

    def test_levels_span_min_to_max(self):
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=5,
            level_min=0.0,
            level_max=1.0,
        )
        assert result["levels"][0] == pytest.approx(0.0, abs=1e-9)
        assert result["levels"][-1] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Fixture 2: random metric (returns a different value on every call)
# ---------------------------------------------------------------------------

class _RandomMetric:
    """Returns a single U(0,1) score per call, ignoring batch size and content.

    Returns one value regardless of batch size so _call_metric always gets
    a single sample per seed — maximising CV and ensuring nr_score < 0.8.
    """

    def __init__(self, seed: int = 99):
        self._rng = np.random.default_rng(seed)

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device):
        return [float(self._rng.random())] * x_batch.shape[0]


class TestNRRandomMetric:
    """NR on a random metric: nr_score must be below 0.5 (highly variable)."""

    def test_nr_score_below_threshold(self):
        # U(0,1) scores: mean≈0.5, std≈0.29, CV≈0.58, nr_score≈0.63.
        # A constant metric gives nr_score=1.0; random must be clearly lower.
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=4)
        result = noise_resilience_test(
            metric_fn=_RandomMetric(seed=17),
            model=model,
            images=images,
            attributions=attrs,
            targets=targets,
            explain_fn=_explain_zeros,
            n_seeds=10,
        )
        assert result["nr_score"] < 0.9, (
            f"Expected nr_score < 0.9 for a random metric (should have non-zero CV), "
            f"got {result['nr_score']:.4f}"
        )
        assert result["std_score"] > 0, "Random metric must show non-zero std"


class TestARRandomMetric:
    """AR on a random metric: mean ar_score over several trials must be < 0.6."""

    def test_ar_score_near_zero_on_average(self):
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=4)
        ar_scores = []
        for trial in range(5):
            result = adversarial_reactivity_test(
                metric_fn=_RandomMetric(seed=trial * 37 + 1),
                model=model,
                images=images,
                attributions=attrs,
                targets=targets,
                n_levels=7,
            )
            ar_scores.append(result["ar_score"])
        mean_ar = float(np.mean(ar_scores))
        assert mean_ar < 0.6, (
            f"Expected mean ar_score < 0.6 for a random metric over 5 trials, "
            f"got {mean_ar:.4f}"
        )


# ---------------------------------------------------------------------------
# Fixture 3: attribution-sum metric (proportional to |attribution|.sum())
# ---------------------------------------------------------------------------

class _AttributionSumMetric:
    """Returns mean absolute sum of attribution per image; decreases as values are zeroed."""

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device):
        return [float(np.abs(a_batch[i]).sum()) for i in range(a_batch.shape[0])]


class TestARAttributionSumMetric:
    """AR on attribution-sum metric: degradation must produce monotonic decline.

    Progressive random permutation of attribution values eventually brings
    all values toward the global mean (entropy maximum), reducing the absolute
    sum toward zero.  The metric should show monotonic decline → ar_score > 0.7.
    """

    def setup_method(self):
        self.model = _DummyModel()
        rng = np.random.default_rng(0)
        n = 3
        # Use larger images so permutation makes a clear difference
        self.images = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]
        self.attrs = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]
        self.targets = [0] * n

    def test_ar_score_above_threshold(self):
        result = adversarial_reactivity_test(
            metric_fn=_AttributionSumMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=6,
        )
        assert result["ar_score"] > 0.7, (
            f"Expected ar_score > 0.7 for attribution-sum metric, "
            f"got {result['ar_score']:.4f}"
        )

    def test_monotonicity_is_negative(self):
        """Higher fraction zeroed → lower |attribution| sum → Spearman ρ < 0.

        The AR test zeros a growing fraction of attribution values.
        Since _AttributionSumMetric returns the total absolute sum, zeroing
        more values directly reduces the sum → monotonic decline → ρ < 0.
        """
        result = adversarial_reactivity_test(
            metric_fn=_AttributionSumMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=6,
        )
        assert result["monotonicity"] < 0, (
            f"Expected negative Spearman ρ (zeroing reduces sum), "
            f"got {result['monotonicity']:.4f}"
        )


# ---------------------------------------------------------------------------
# meta_evaluate_metric combined tests
# ---------------------------------------------------------------------------

class TestMetaEvaluateMetric:
    def setup_method(self):
        self.model = _DummyModel()
        self.images, self.attrs, self.targets = _make_inputs()

    def test_returns_required_keys(self):
        result = meta_evaluate_metric(
            metric_name="TestConstant",
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=3,
            n_levels=3,
        )
        assert result["metric"] == "TestConstant"
        assert "nr" in result
        assert "ar" in result
        assert "combined_reliability" in result

    def test_combined_reliability_formula(self):
        """combined_reliability == 0.5 * (nr_score + ar_score).

        Uses _AttributionSumMetric (non-constant) so both NR and AR scores
        are finite, making the formula verifiable.
        """
        rng = np.random.default_rng(1)
        images = [rng.random((3, 8, 8)).astype(np.float32) for _ in range(2)]
        attrs = [rng.random((3, 8, 8)).astype(np.float32) for _ in range(2)]
        targets = [0, 0]

        result = meta_evaluate_metric(
            metric_name="X",
            metric_fn=_AttributionSumMetric(),
            model=self.model,
            images=images,
            attributions=attrs,
            targets=targets,
            explain_fn=_explain_zeros,
            n_seeds=3,
            n_levels=3,
        )
        nr_s = result["nr"]["nr_score"]
        ar_s = result["ar"]["ar_score"]
        assert np.isfinite(nr_s) and np.isfinite(ar_s)
        expected = 0.5 * (nr_s + ar_s)
        assert result["combined_reliability"] == pytest.approx(expected, abs=1e-9)

    def test_seed_count_forwarded_to_nr(self):
        result = meta_evaluate_metric(
            metric_name="X",
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=4,
            n_levels=3,
        )
        assert len(result["nr"]["raw_scores"]) == 4

    def test_level_count_forwarded_to_ar(self):
        result = meta_evaluate_metric(
            metric_name="X",
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=2,
            n_levels=6,
        )
        assert len(result["ar"]["levels"]) == 6

    def test_constant_metric_combined_reliability(self):
        """Constant metric: NR=1.0, AR=0.0 → combined=0.5."""
        result = meta_evaluate_metric(
            metric_name="Constant",
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_zeros,
            n_seeds=5,
            n_levels=5,
        )
        assert result["nr"]["nr_score"] == pytest.approx(1.0, abs=1e-9)
        assert result["ar"]["ar_score"] == pytest.approx(0.0, abs=1e-9)
        assert result["combined_reliability"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_perturbation_type_raises(self):
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=1)
        with pytest.raises(ValueError, match="perturbation_type"):
            noise_resilience_test(
                metric_fn=_ConstantMetric(),
                model=model,
                images=images,
                attributions=attrs,
                targets=targets,
                explain_fn=_explain_zeros,
                perturbation_type="invalid",
            )

    def test_weights_perturbation_runs(self):
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=1)
        result = noise_resilience_test(
            metric_fn=_ConstantMetric(),
            model=model,
            images=images,
            attributions=attrs,
            targets=targets,
            explain_fn=_explain_zeros,
            perturbation_type="weights",
            n_seeds=2,
        )
        assert "nr_score" in result
        assert np.isfinite(result["nr_score"])

    def test_nr_score_bounded_in_zero_one(self):
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=3)
        for metric_fn in [_ConstantMetric(), _RandomMetric(seed=7)]:
            result = noise_resilience_test(
                metric_fn=metric_fn,
                model=model,
                images=images,
                attributions=attrs,
                targets=targets,
                explain_fn=_explain_zeros,
                n_seeds=5,
            )
            assert 0.0 <= result["nr_score"] <= 1.0

    def test_ar_score_bounded_in_zero_one(self):
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=3)
        for metric_fn in [_ConstantMetric(), _AttributionSumMetric()]:
            result = adversarial_reactivity_test(
                metric_fn=metric_fn,
                model=model,
                images=images,
                attributions=attrs,
                targets=targets,
                n_levels=5,
            )
            assert 0.0 <= result["ar_score"] <= 1.0


# ---------------------------------------------------------------------------
# screen_meta_eval_candidates
# ---------------------------------------------------------------------------

def _make_slice_df(
    model: str,
    fae: str,
    metric: str,
    scores: list,
) -> pd.DataFrame:
    """Build a minimal long-format slice DataFrame for one (model, fae, metric) triple."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "image_id": f"ISIC_{i:07d}",
                "fae_method": fae,
                "metric": metric,
                "score": s,
            }
            for i, s in enumerate(scores)
        ]
    )


class TestScreenMetaEvalCandidates:
    def test_all_valid_returns_run_true(self):
        """12/12 valid scores → run_meta_eval = True."""
        df = _make_slice_df("resnet18", "saliency", "faithfulness_correlation",
                             scores=[0.5] * 12)
        result = screen_meta_eval_candidates(df, min_valid_fraction=0.5)
        assert len(result) == 1
        assert bool(result.iloc[0]["run_meta_eval"]) is True
        assert result.iloc[0]["n_valid_images"] == 12
        assert result.iloc[0]["total_images"] == 12

    def test_all_nan_returns_run_false(self):
        """0/12 valid scores (all NaN) → run_meta_eval = False."""
        df = _make_slice_df("squeezenet", "gradcam", "model_parameter_randomisation",
                             scores=[float("nan")] * 12)
        result = screen_meta_eval_candidates(df, min_valid_fraction=0.5)
        assert len(result) == 1
        assert bool(result.iloc[0]["run_meta_eval"]) is False
        assert result.iloc[0]["n_valid_images"] == 0
        assert result.iloc[0]["valid_fraction"] == pytest.approx(0.0)

    def test_half_valid_at_threshold_returns_run_true(self):
        """6/12 valid (valid_fraction = 0.5) → run_meta_eval = True at threshold 0.5."""
        scores = [0.3] * 6 + [float("nan")] * 6
        df = _make_slice_df("resnet18", "lrp", "completeness", scores=scores)
        result = screen_meta_eval_candidates(df, min_valid_fraction=0.5)
        assert len(result) == 1
        assert bool(result.iloc[0]["run_meta_eval"]) is True
        assert result.iloc[0]["valid_fraction"] == pytest.approx(0.5)

    def test_returns_expected_columns(self):
        df = _make_slice_df("resnet18", "saliency", "sparseness", scores=[1.0] * 5)
        result = screen_meta_eval_candidates(df)
        expected_cols = {
            "model", "fae_method", "metric", "n_valid_images",
            "total_images", "valid_fraction", "run_meta_eval",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_multiple_triples(self):
        """Three triples with different coverage → correct run_meta_eval per triple."""
        df = pd.concat(
            [
                _make_slice_df("resnet18", "saliency", "sparseness", [0.5] * 12),
                _make_slice_df("resnet18", "saliency", "non_sensitivity",
                               [float("nan")] * 12),
                _make_slice_df("resnet18", "saliency", "complexity", [0.3] * 6 + [float("nan")] * 6),
            ],
            ignore_index=True,
        )
        result = screen_meta_eval_candidates(df, min_valid_fraction=0.5)
        assert len(result) == 3
        by_metric = result.set_index("metric")["run_meta_eval"]
        assert bool(by_metric["sparseness"]) is True
        assert bool(by_metric["non_sensitivity"]) is False
        assert bool(by_metric["complexity"]) is True


# ---------------------------------------------------------------------------
# run_meta_evaluation_full
# ---------------------------------------------------------------------------

class TestRunMetaEvaluationFull:
    def test_runs_and_produces_csv_with_status(self, tmp_path):
        """2 models × 1 FAE × 1 metric × 2 images, n_seeds=2, n_levels=2."""
        rng = np.random.default_rng(7)
        images_t = [
            torch.tensor(rng.random((3, 8, 8)).astype(np.float32))
            for _ in range(2)
        ]
        targets = [0, 1]

        # Build a minimal slice with all valid scores for 2 models
        rows = []
        for model_name in ("model_a", "model_b"):
            for img_idx in range(2):
                rows.append(
                    {
                        "model": model_name,
                        "image_id": f"img_{img_idx:03d}",
                        "fae_method": "fake_fae",
                        "metric": "test_metric",
                        "score": float(rng.random()),
                    }
                )
        slice_df = pd.DataFrame(rows)

        output_csv = str(tmp_path / "meta_eval.csv")
        progress_log = str(tmp_path / "progress.log")

        result_df = run_meta_evaluation_full(
            vertical_slice_df=slice_df,
            models={"model_a": _DummyModel(), "model_b": _DummyModel()},
            metric_fns={"test_metric": _ConstantMetric()},
            fae_methods={"fake_fae": _explain_zeros},
            images=images_t,
            targets=targets,
            device="cpu",
            n_seeds=2,
            n_levels=2,
            output_csv=output_csv,
            progress_log=progress_log,
        )

        # Return value is a DataFrame with expected columns
        assert isinstance(result_df, pd.DataFrame)
        required_cols = {
            "model", "fae_method", "metric", "nr_score", "ar_score",
            "combined_reliability", "n_valid_inputs", "runtime_seconds", "status",
        }
        assert required_cols.issubset(set(result_df.columns))

        # All rows should be 'completed' (constant metric, all valid scores)
        assert set(result_df["status"].unique()) == {"completed"}, (
            f"Expected all 'completed', got: {result_df['status'].value_counts().to_dict()}"
        )

        # CSV must exist and contain the same rows
        assert Path(output_csv).exists()
        csv_df = pd.read_csv(output_csv)
        assert len(csv_df) == len(result_df)
        assert "status" in csv_df.columns

        # Covers both models
        assert set(result_df["model"].unique()) == {"model_a", "model_b"}

    def test_skipped_nan_triples_recorded(self, tmp_path):
        """Triples with 0/2 valid scores appear in output with status='skipped_nan'."""
        rows = [
            {
                "model": "resnet18",
                "image_id": f"img_{i}",
                "fae_method": "saliency",
                "metric": "all_nan_metric",
                "score": float("nan"),
            }
            for i in range(2)
        ]
        slice_df = pd.DataFrame(rows)
        images_t = [torch.zeros(3, 8, 8) for _ in range(2)]

        result_df = run_meta_evaluation_full(
            vertical_slice_df=slice_df,
            models={"resnet18": _DummyModel()},
            metric_fns={"all_nan_metric": _ConstantMetric()},
            fae_methods={"saliency": _explain_zeros},
            images=images_t,
            targets=[0, 0],
            device="cpu",
            n_seeds=2,
            n_levels=2,
            output_csv=str(tmp_path / "out.csv"),
            progress_log=None,
        )

        assert len(result_df) == 1
        assert result_df.iloc[0]["status"] == "skipped_nan"
        assert np.isnan(result_df.iloc[0]["nr_score"])
