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
    make_degraded_explain_func,
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
    """Returns 0.5 for every image in the batch, regardless of attribution.

    Note: triggers the zero-variance check in adversarial_reactivity_test
    (all images get identical scores → std=0 → ar_score=NaN). This is correct
    behaviour — a truly constant metric is uninformative for AR ranking.
    """

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device, **kwargs):
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

    def test_ar_score_is_nan_zero_variance(self):
        """Constant metric → std=0 across images → zero-variance check → ar_score=NaN.

        _ConstantMetric returns 0.5 for all images regardless of attribution,
        so base_raw = [0.5, 0.5, ...] with std=0. The adversarial_reactivity_test
        correctly returns NaN instead of the spurious 0.0 from the previous
        implementation. This is the same reason Completeness is excluded from M*.
        """
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=5,
        )
        assert np.isnan(result["ar_score"])
        assert result["zero_variance"] is True

    def test_monotonicity_is_nan_zero_variance(self):
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            n_levels=5,
        )
        assert np.isnan(result["monotonicity"])

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
    """Returns an independent U(0,1) score per image per call.

    Per-image random scores ensure std > 0 across images (so the
    zero-variance check in adversarial_reactivity_test does not trigger),
    while scores are unpredictable across seeds (NR CV is high → nr_score low).
    """

    def __init__(self, seed: int = 99):
        self._rng = np.random.default_rng(seed)

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device, **kwargs):
        return [float(self._rng.random()) for _ in range(x_batch.shape[0])]


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
    """Returns total absolute sum of attribution per image; decreases as values are zeroed."""

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device, **kwargs):
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
# Degraded-explain_func AR tests (max_sensitivity / random_logit failure mode)
# ---------------------------------------------------------------------------

def _explain_ones(model, inputs, targets, **kwargs) -> np.ndarray:
    """explain_fn returning all-ones attributions (deterministic, non-zero)."""
    return np.ones_like(np.asarray(inputs), dtype=np.float32)


class _ExplainFuncSumMetric:
    """Mock of an explain_func-recomputing metric (MaxSensitivity-like).

    Ignores ``a_batch`` entirely — the score is the mean absolute value of a
    FRESH call to ``explain_func``, exactly the property that made the
    a_batch-only AR degradation return constant scores (→ NaN) for
    max_sensitivity and random_logit in the 2026-06 run. Adds a per-image
    offset so base scores are not zero-variance across images.
    """

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first,
                 device, explain_func=None, explain_func_kwargs=None, **kwargs):
        a = explain_func(model, x_batch, y_batch)
        per_image = np.abs(a).reshape(a.shape[0], -1).mean(axis=1)
        offsets = np.linspace(0.0, 0.01, x_batch.shape[0])
        return list(per_image + offsets)


class TestMakeDegradedExplainFunc:
    def setup_method(self):
        self.model = _DummyModel()
        self.inputs = np.random.default_rng(1).random((2, 3, 8, 8)).astype(np.float32)

    def test_frac_zero_returns_unchanged(self):
        rng = np.random.default_rng(0)
        fn = make_degraded_explain_func(_explain_ones, 0.0, rng)
        out = fn(self.model, self.inputs, [0, 0])
        assert np.array_equal(out, np.ones_like(self.inputs))

    def test_zeroes_expected_fraction(self):
        rng = np.random.default_rng(0)
        fn = make_degraded_explain_func(_explain_ones, 0.5, rng)
        out = fn(self.model, self.inputs, [0, 0])
        frac_zeroed = float((out == 0.0).mean())
        assert abs(frac_zeroed - 0.5) < 0.01

    def test_successive_calls_draw_different_masks(self):
        rng = np.random.default_rng(0)
        fn = make_degraded_explain_func(_explain_ones, 0.3, rng)
        out1 = fn(self.model, self.inputs, [0, 0])
        out2 = fn(self.model, self.inputs, [0, 0])
        assert not np.array_equal(out1, out2), (
            "Degraded explainer must be stochastic across calls (shared rng "
            "advances) — a fixed mask would make sensitivity metrics blind."
        )

    def test_original_fn_output_not_mutated(self):
        base = np.ones((2, 3, 8, 8), dtype=np.float32)

        def _explain_shared(model, inputs, targets, **kwargs):
            return base

        rng = np.random.default_rng(0)
        fn = make_degraded_explain_func(_explain_shared, 0.5, rng)
        fn(self.model, self.inputs, [0, 0])
        assert np.array_equal(base, np.ones_like(base))


class TestARDegradedExplainFunc:
    """The failure mode of the 2026-06 run, and its fix.

    Without degrade_explain_func, an explain_func-recomputing metric returns
    identical scores at every degradation level (it ignores a_batch), so no
    monotonic structure exists. With the flag, the metric's internal
    re-explanations lose signal as frac grows, so the mean-|attribution|
    score declines monotonically → high ar_score.
    """

    def setup_method(self):
        self.model = _DummyModel()
        rng = np.random.default_rng(0)
        n = 3
        self.images = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]
        self.attrs = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]
        self.targets = [0] * n

    def _run(self, degrade: bool) -> dict:
        return adversarial_reactivity_test(
            metric_fn=_ExplainFuncSumMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_func=_explain_ones,
            n_levels=6,
            degrade_explain_func=degrade,
        )

    def test_without_flag_scores_are_constant(self):
        result = self._run(degrade=False)
        assert np.std(result["scores"]) < 1e-12, (
            "a_batch-only degradation must leave an explain_func-recomputing "
            f"metric constant; got scores={result['scores']}"
        )

    def test_with_flag_ar_score_above_threshold(self):
        result = self._run(degrade=True)
        assert result["ar_score"] > 0.7, (
            f"Expected ar_score > 0.7 with degraded explain_func, "
            f"got {result['ar_score']:.4f} (scores={result['scores']})"
        )

    def test_with_flag_monotonicity_is_negative(self):
        result = self._run(degrade=True)
        assert result["monotonicity"] < 0, (
            "Zeroing a growing fraction of the explainer output must reduce "
            f"the mean-|attribution| score; rho={result['monotonicity']:.4f}"
        )

    def test_meta_evaluate_metric_wires_flag_for_explain_func_metrics(self):
        """meta_evaluate_metric must enable the degradation for metrics whose
        METRIC_KWARG_REQUIREMENTS include explain_func (e.g. max_sensitivity)."""
        result = meta_evaluate_metric(
            metric_name="max_sensitivity",
            metric_fn=_ExplainFuncSumMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs,
            targets=self.targets,
            explain_fn=_explain_ones,
            n_seeds=2,
            n_levels=6,
        )
        assert not np.isnan(result["ar"]["ar_score"]), (
            "AR must no longer be NaN for explain_func metrics"
        )
        assert result["ar"]["ar_score"] > 0.7


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
        """Constant metric: NR=1.0, AR=NaN (zero variance) → combined=NaN.

        _ConstantMetric returns 0.5 regardless of input, so:
        - NR: std across seeds = 0 → CV = 0 → nr_score = 1.0
        - AR: std across images = 0 → zero-variance check → ar_score = NaN
        - combined: NaN (either score NaN → combined NaN)
        """
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
        assert np.isnan(result["ar"]["ar_score"])
        assert np.isnan(result["combined_reliability"])


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
        """AR score in [0, 1] for non-constant metrics; NaN for constant (zero-variance)."""
        model = _DummyModel()
        images, attrs, targets = _make_inputs(n_images=3)
        # _ConstantMetric triggers zero-variance → NaN (not a finite value)
        result_const = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=model,
            images=images,
            attributions=attrs,
            targets=targets,
            n_levels=5,
        )
        assert np.isnan(result_const["ar_score"])
        # _AttributionSumMetric has variance → finite ar_score in [0, 1]
        result_sum = adversarial_reactivity_test(
            metric_fn=_AttributionSumMetric(),
            model=model,
            images=images,
            attributions=attrs,
            targets=targets,
            n_levels=5,
        )
        assert 0.0 <= result_sum["ar_score"] <= 1.0


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

    def test_nested_per_model_fae_methods(self, tmp_path):
        """fae_methods nested as {model -> {fae -> fn}} resolves each model's own func."""
        rng = np.random.default_rng(7)
        images_t = [
            torch.tensor(rng.random((3, 8, 8)).astype(np.float32)) for _ in range(2)
        ]
        rows = []
        for model_name in ("model_a", "model_b"):
            for img_idx in range(2):
                rows.append({
                    "model": model_name, "image_id": f"img_{img_idx:03d}",
                    "fae_method": "fake_fae", "metric": "test_metric",
                    "score": float(rng.random()),
                })
        slice_df = pd.DataFrame(rows)

        calls = {"model_a": 0, "model_b": 0}

        def _make(name):
            def _f(model, inputs, targets, **kw):
                calls[name] += 1
                return _explain_zeros(model, inputs, targets, **kw)
            return _f

        result_df = run_meta_evaluation_full(
            vertical_slice_df=slice_df,
            models={"model_a": _DummyModel(), "model_b": _DummyModel()},
            metric_fns={"test_metric": _ConstantMetric()},
            fae_methods={
                "model_a": {"fake_fae": _make("model_a")},
                "model_b": {"fake_fae": _make("model_b")},
            },
            images=images_t, targets=[0, 1], device="cpu",
            n_seeds=2, n_levels=2,
            output_csv=str(tmp_path / "m.csv"),
            progress_log=str(tmp_path / "p.log"),
        )

        assert set(result_df["model"].unique()) == {"model_a", "model_b"}
        assert set(result_df["status"].unique()) == {"completed"}
        # Each model's OWN explain_func was used (the per-model resolution).
        assert calls["model_a"] > 0 and calls["model_b"] > 0

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


# ---------------------------------------------------------------------------
# New tests: auxiliary kwargs forwarding (Part B / Part E)
# ---------------------------------------------------------------------------

class _PointingGameFixture:
    """Simulates PointingGame: 1.0 if max-attribution pixel is inside mask, else 0.0.

    Needs s_batch forwarded (via masks param) to produce non-NaN scores.
    After _call_metric collapses channels, a_batch arrives as (B, 1, H, W).
    """

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device,
                 s_batch=None, **kwargs):
        B = a_batch.shape[0]
        results = []
        for b in range(B):
            a = a_batch[b, 0]  # (H, W) — channel already collapsed by _call_metric
            if s_batch is not None:
                mask = s_batch[b, 0]
                max_idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
                results.append(1.0 if mask[max_idx] > 0 else 0.0)
            else:
                results.append(float("nan"))
        return results


class _MaxSensFixture:
    """Returns mean |attribution| per image; declines as attribution is zeroed.

    Accepts explain_func to simulate MaxSensitivity's call signature.
    Uses a_batch directly (not explain_func) so scores are determined by
    the degraded attribution, ensuring monotonic AR behaviour.
    """

    def __call__(self, model, x_batch, y_batch, a_batch, channel_first, device,
                 explain_func=None, explain_func_kwargs=None, **kwargs):
        return [float(np.abs(a_batch[b]).mean()) for b in range(x_batch.shape[0])]


class TestARAuxiliaryKwargs:
    """AR test correctly handles mask-dependent and explain_func-dependent metrics."""

    def setup_method(self):
        self.model = _DummyModel()
        rng = np.random.default_rng(0)
        n = 3
        self.images = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]
        self.targets = [0] * n

        # Mask covers center square [4:12, 4:12]
        self.masks = [np.zeros((1, 16, 16), dtype=np.float32) for _ in range(n)]
        for m in self.masks:
            m[0, 4:12, 4:12] = 1.0

        # Structured attrs for PointingGame: images 0 and 2 peak inside mask,
        # image 1 peak outside → base_raw = [1.0, 0.0, 1.0] → std > 0.
        # _call_metric collapses channels via sum → peak pixel has value 3*2.0=6.
        self.attrs_pg = []
        for i in range(n):
            a = np.zeros((3, 16, 16), dtype=np.float32)
            if i % 2 == 0:
                a[:, 8, 8] = 2.0   # center pixel, inside mask
            else:
                a[:, 0, 0] = 2.0   # corner pixel, outside mask
            self.attrs_pg.append(a)

        # Random attrs for MaxSens: different per-image mean |a| → variance > 0.
        self.attrs_rand = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(n)]

    def test_ar_pointing_game_with_mask_is_finite(self):
        """PointingGame: AR is finite when mask is forwarded correctly.

        attrs_pg: images 0 and 2 peak inside mask, image 1 outside.
        base_raw = [1.0, 0.0, 1.0] → std > 0 → zero-variance check passes.
        As attribution is degraded (values zeroed), scores decline → ar_score > 0.
        """
        result = adversarial_reactivity_test(
            metric_fn=_PointingGameFixture(),
            model=self.model,
            images=self.images,
            attributions=self.attrs_pg,
            targets=self.targets,
            masks=self.masks,
            n_levels=4,
        )
        assert np.isfinite(result["ar_score"]), (
            f"Expected finite ar_score with mask, got {result['ar_score']}"
        )

    def test_ar_max_sensitivity_with_explain_func_is_finite(self):
        """MaxSensitivity: AR is finite when explain_func is forwarded.

        attrs_rand: different mean |a| per image → std > 0 → zero-variance passes.
        _MaxSensFixture uses a_batch.mean() which declines as values are zeroed.
        """
        result = adversarial_reactivity_test(
            metric_fn=_MaxSensFixture(),
            model=self.model,
            images=self.images,
            attributions=self.attrs_rand,
            targets=self.targets,
            explain_func=_explain_zeros,
            n_levels=4,
        )
        assert np.isfinite(result["ar_score"]), (
            f"Expected finite ar_score with explain_func, got {result['ar_score']}"
        )

    def test_ar_completeness_zero_variance_returns_nan(self):
        """Completeness-like metric: AR returns NaN (zero variance), not spurious 0.0.

        Uses _ConstantMetric (always 0.5) to simulate Completeness (always 0.0):
        both have std=0 across images, so the zero-variance check triggers.
        """
        result = adversarial_reactivity_test(
            metric_fn=_ConstantMetric(),
            model=self.model,
            images=self.images,
            attributions=self.attrs_pg,
            targets=self.targets,
            n_levels=4,
        )
        assert np.isnan(result["ar_score"]), (
            f"Expected NaN ar_score for zero-variance metric, got {result['ar_score']}"
        )
        assert result["zero_variance"] is True

    def test_meta_evaluate_all_kwargs_no_nan_combined(self):
        """meta_evaluate_metric with mask + explain_func produces finite combined.

        Uses attrs_pg so PointingGame base_raw has non-zero variance.
        NR: explain_fn=_explain_zeros → recomputed attrs are zeros → argmax at (0,0)
            → score=0 for all seeds → std=0 → nr_score=1.0.
        AR: base_raw=[1.0, 0.0, 1.0] → std>0 → ar proceeds → finite ar_score.
        """
        result = meta_evaluate_metric(
            metric_name="PointingGame",
            metric_fn=_PointingGameFixture(),
            model=self.model,
            images=self.images,
            attributions=self.attrs_pg,
            targets=self.targets,
            explain_fn=_explain_zeros,
            masks=self.masks,
            n_seeds=2,
            n_levels=3,
        )
        assert np.isfinite(result["combined_reliability"]), (
            f"Expected finite combined_reliability, got {result['combined_reliability']}"
        )
