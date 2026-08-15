"""Tests for src/aggregation/weighting.py.

Part A tests (autoweighted_weights) — active.
Part B tests (metaquantus_discounted_weights) — skipped pending
    results/meta_evaluation_reliability.csv from the Colab AR-fixed run.
Integration test — autoweighted only on vertical_slice_7fae_12metrics.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aggregation.weighting import autoweighted_weights, metaquantus_discounted_weights

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
_VERTICAL_SLICE = _RESULTS_DIR / "vertical_slice_7fae_12metrics.csv"
_RELIABILITY_CSV = _RESULTS_DIR / "meta_evaluation_reliability.csv"


def _make_scores_df(
    metric_scores: dict[str, list[float]],
    model: str = "resnet18",
    fae_method: str = "gradcam",
) -> pd.DataFrame:
    """Build a minimal long-format scores DataFrame from {metric: [scores]}."""
    rows = []
    for metric, scores in metric_scores.items():
        for i, s in enumerate(scores):
            rows.append(
                {
                    "model": model,
                    "image_id": f"img_{i:03d}",
                    "fae_method": fae_method,
                    "metric": metric,
                    "score": s,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part A: autoweighted_weights
# ---------------------------------------------------------------------------


class TestAutoweighted:
    def test_uniform_cv_gives_uniform_weights(self):
        """Metrics with identical CV must receive equal weights (1/k each).

        All three metrics share the same score array → identical mean, std,
        CV, and raw_weight → each normalised weight = 1/3.
        """
        scores = list(np.linspace(0.5, 1.5, 12))  # mean=1.0, std≠0
        df = _make_scores_df({"mA": scores, "mB": scores, "mC": scores})
        weights = autoweighted_weights(df)

        assert len(weights) == 1
        w = next(iter(weights.values()))

        assert set(w.keys()) == {"mA", "mB", "mC"}
        for metric, wt in w.items():
            assert wt == pytest.approx(1.0 / 3.0, rel=1e-9), (
                f"Expected uniform weight 1/3 for {metric}, got {wt}"
            )

    def test_low_cv_metric_gets_higher_weight(self):
        """A metric with near-zero variance must dominate the weight vector.

        mA: scores all equal 1.0  → std=0, cv=0, raw_weight = 1/ε  (huge)
        mB: linearly spaced scores → std>0, cv>0, raw_weight  ≪ 1/ε
        mC: same as mB
        """
        n = 12
        scores_const = [1.0] * n
        scores_noisy = list(np.linspace(0.1, 2.0, n))

        df = _make_scores_df(
            {"mA": scores_const, "mB": scores_noisy, "mC": scores_noisy}
        )
        weights = autoweighted_weights(df)
        w = next(iter(weights.values()))

        assert w["mA"] > w["mB"], "low-CV metric must outweigh high-CV metric"
        assert w["mA"] > w["mC"], "low-CV metric must outweigh high-CV metric"

    def test_all_nan_metric_gets_zero_weight(self):
        """A metric with all-NaN scores must receive weight 0; others sum to 1.

        mC is entirely NaN.  After normalisation, mA and mB share all the
        weight and mC is exactly 0.
        """
        scores = list(np.linspace(0.5, 1.5, 10))
        nan_scores = [float("nan")] * 10

        df = _make_scores_df(
            {"mA": scores, "mB": scores, "mC": nan_scores}
        )
        weights = autoweighted_weights(df)
        w = next(iter(weights.values()))

        assert w["mC"] == pytest.approx(0.0), "all-NaN metric must have weight 0"
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"Remaining weights must still sum to 1.0, got {total}"
        )

    def test_weights_sum_to_one(self):
        """Sanity: weights always sum to 1 for valid input."""
        rng = np.random.default_rng(42)
        metrics = {f"m{i}": list(rng.uniform(0.1, 1.0, 15)) for i in range(6)}
        df = _make_scores_df(metrics)
        weights = autoweighted_weights(df)
        w = next(iter(weights.values()))
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_group_by_model_produces_separate_weight_vectors(self):
        """Two models must each get an independent weight vector."""
        scores = list(np.linspace(0.5, 1.5, 8))
        rows = []
        for model in ("resnet18", "squeezenet"):
            for metric, vals in {"mA": scores, "mB": scores[::-1]}.items():
                for i, s in enumerate(vals):
                    rows.append(
                        {
                            "model": model,
                            "image_id": f"img_{i:03d}",
                            "fae_method": "gradcam",
                            "metric": metric,
                            "score": s,
                        }
                    )
        df = pd.DataFrame(rows)
        weights = autoweighted_weights(df, group_by=("model",))

        assert ("resnet18",) in weights
        assert ("squeezenet",) in weights
        for w in weights.values():
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_explicit_metric_set_respected(self):
        """Only metrics in metric_set must appear in the output."""
        scores = list(np.linspace(0.2, 0.8, 8))
        df = _make_scores_df({"mA": scores, "mB": scores, "mC": scores})
        weights = autoweighted_weights(df, metric_set=["mA", "mC"])
        w = next(iter(weights.values()))
        assert set(w.keys()) == {"mA", "mC"}
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Part B: metaquantus_discounted_weights
# ---------------------------------------------------------------------------


class TestMetaquantusDiscount:
    def test_uniform_reliability_equals_autoweighted(self):
        """Uniform reliability (all = 0.5) must cancel in normalisation.

        w_k = (AW_k × 0.5) / Σ(AW_j × 0.5) = AW_k / Σ AW_j = AW_k.
        Result must be identical to plain autoweighted_weights.
        """
        scores = list(np.linspace(0.5, 1.5, 12))
        df = _make_scores_df({"mA": scores, "mB": list(reversed(scores)), "mC": scores})

        rel_rows = []
        for fae in ("gradcam", "saliency"):
            for metric in ("mA", "mB", "mC"):
                rel_rows.append(
                    {
                        "model": "resnet18",
                        "fae_method": fae,
                        "metric": metric,
                        "combined_reliability": 0.5,
                    }
                )
        rel_df = pd.DataFrame(rel_rows)

        aw = autoweighted_weights(df)
        mq = metaquantus_discounted_weights(df, rel_df)

        aw_w = next(iter(aw.values()))
        mq_w = next(iter(mq.values()))
        for m in aw_w:
            assert mq_w[m] == pytest.approx(aw_w[m], rel=1e-6), (
                f"Uniform reliability should not change weights; metric {m}: "
                f"AW={aw_w[m]:.6f}, MQ={mq_w[m]:.6f}"
            )

    def test_zero_reliability_metric_gets_zero_weight(self):
        """A metric with reliability=0.0 must receive weight 0 after discount."""
        scores = list(np.linspace(0.5, 1.5, 10))
        df = _make_scores_df({"mA": scores, "mB": scores, "mC": scores})

        rel_rows = [
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mA",
             "combined_reliability": 0.8},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mB",
             "combined_reliability": 0.7},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mC",
             "combined_reliability": 0.0},
        ]
        rel_df = pd.DataFrame(rel_rows)

        weights = metaquantus_discounted_weights(df, rel_df)
        w = next(iter(weights.values()))

        assert w["mC"] == pytest.approx(0.0), "reliability=0.0 must produce weight 0"
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_nan_reliability_uses_category_fallback(self):
        """NaN combined_reliability (no nr_score) → category mean (fallback step 2).

        mA and mB are in the same category 'Cat1'.
        mB has combined_reliability=NaN and no nr_score column.
        Expected fallback for mB: 0.8 (mean of Cat1 peer mA).
        """
        scores = list(np.linspace(0.5, 1.5, 10))
        df = _make_scores_df({"mA": scores, "mB": scores, "mC": scores})

        rel_rows = [
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mA",
             "combined_reliability": 0.8},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mB",
             "combined_reliability": float("nan")},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mC",
             "combined_reliability": 0.6},
        ]
        rel_df = pd.DataFrame(rel_rows)

        category_map = {"mA": "Cat1", "mB": "Cat1", "mC": "Cat2"}

        weights = metaquantus_discounted_weights(df, rel_df, category_map=category_map)
        w = next(iter(weights.values()))

        # mB discount = 0.8 (fallback from mA in Cat1).
        # Uniform AW → after discount: mA→0.8, mB→0.8, mC→0.6.
        # Normalised: mA and mB equal weight, mC smaller.
        assert w["mA"] == pytest.approx(w["mB"], rel=1e-6), (
            "mA and mB have same AW and same reliability after fallback"
        )
        assert w["mC"] < w["mA"], "mC reliability 0.6 < 0.8 → lower weight"
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_mq_discount_nr_only_fallback(self):
        """combined_reliability=NaN but nr_score=0.7 → λ=0.7 (NR-only, fallback step 1).

        Must use NR-only before falling through to category or global mean.
        mA and mC have finite combined_reliability=0.9; global mean = 0.9.
        mB has combined=NaN, nr=0.7.  NR-only must win over global mean (0.9).
        """
        scores = list(np.linspace(0.5, 1.5, 10))
        df = _make_scores_df({"mA": scores, "mB": scores, "mC": scores})

        rel_rows = [
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mA",
             "combined_reliability": 0.9, "nr_score": 0.95},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mB",
             "combined_reliability": float("nan"), "nr_score": 0.7},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mC",
             "combined_reliability": 0.9, "nr_score": 0.85},
        ]
        rel_df = pd.DataFrame(rel_rows)

        # No category_map → category fallback (step 2) cannot fire.
        weights, sources = metaquantus_discounted_weights(df, rel_df, return_sources=True)
        w = next(iter(weights.values()))
        src = next(iter(sources.values()))

        assert src["mB"] == "nr_only", (
            f"Expected NR-only fallback for mB, got '{src['mB']}'"
        )
        # Uniform AW; discounts: mA=0.9, mB=0.7, mC=0.9 → total=2.5.
        assert w["mB"] == pytest.approx(0.7 / (0.9 + 0.7 + 0.9), rel=1e-6), (
            f"NR-only fallback with nr_score=0.7 → λ=0.7; got {w['mB']:.6f}"
        )
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_nan_reliability_falls_back_to_autoweighted(self):
        """When ALL combined_reliability values are NaN, discount = 1.0 everywhere.

        Result must equal plain autoweighted_weights.
        """
        scores_A = list(np.linspace(0.5, 1.5, 10))
        scores_B = list(np.linspace(0.2, 0.8, 10))
        df = _make_scores_df({"mA": scores_A, "mB": scores_B})

        rel_rows = [
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mA",
             "combined_reliability": float("nan")},
            {"model": "resnet18", "fae_method": "gradcam", "metric": "mB",
             "combined_reliability": float("nan")},
        ]
        rel_df = pd.DataFrame(rel_rows)

        aw = autoweighted_weights(df)
        mq = metaquantus_discounted_weights(df, rel_df)

        aw_w = next(iter(aw.values()))
        mq_w = next(iter(mq.values()))
        for m in aw_w:
            assert mq_w[m] == pytest.approx(aw_w[m], rel=1e-6), (
                f"All-NaN reliability → discount=1 → must equal AW; "
                f"metric {m}: AW={aw_w[m]:.6f}, MQ={mq_w[m]:.6f}"
            )


# ---------------------------------------------------------------------------
# Integration: autoweighted on real vertical-slice data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _VERTICAL_SLICE.exists(),
    reason=f"vertical_slice_7fae_12metrics.csv not found at {_VERTICAL_SLICE}",
)
class TestIntegrationAutoweighted:
    def test_weights_sum_to_one_per_model(self):
        """Autoweighted weights must sum to 1.0 per model on real data."""
        df = pd.read_csv(_VERTICAL_SLICE)
        # Use a small sample to keep the test fast: 2 image_ids per model.
        sampled_images = (
            df.groupby("model")["image_id"]
            .apply(lambda x: x.drop_duplicates().iloc[:2])
            .reset_index(drop=True)
        )
        df_sample = df[df["image_id"].isin(sampled_images)]

        weights = autoweighted_weights(df_sample, group_by=("model",))

        assert len(weights) > 0, "Expected at least one group in real data"
        for group_key, w in weights.items():
            total = sum(w.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Weights do not sum to 1 for group {group_key}: {total}"
            )

    def test_non_negative_weights(self):
        """All weights must be >= 0."""
        df = pd.read_csv(_VERTICAL_SLICE)
        weights = autoweighted_weights(df, group_by=("model",))
        for group_key, w in weights.items():
            for metric, wt in w.items():
                assert wt >= 0.0, (
                    f"Negative weight for group {group_key}, metric {metric}: {wt}"
                )

    def test_all_nan_metrics_get_zero_weight(self):
        """non_sensitivity (168/168 NaN) must get weight 0 for all models."""
        df = pd.read_csv(_VERTICAL_SLICE)
        weights = autoweighted_weights(df, group_by=("model",))
        for group_key, w in weights.items():
            if "non_sensitivity" in w:
                assert w["non_sensitivity"] == pytest.approx(0.0, abs=1e-12), (
                    f"non_sensitivity is 100% NaN but got weight "
                    f"{w['non_sensitivity']} for {group_key}"
                )


# ---------------------------------------------------------------------------
# Part D: metaquantus_discounted_weights on real data
# ---------------------------------------------------------------------------

_QUANTUS_CATEGORIES = {
    "avg_sensitivity": "robustness",
    "max_sensitivity": "robustness",
    "completeness": "faithfulness",
    "faithfulness_correlation": "faithfulness",
    "pixel_flipping": "faithfulness",
    "pointing_game": "faithfulness",
    "relevance_mass_accuracy": "faithfulness",
    "non_sensitivity": "faithfulness",
    "complexity": "complexity",
    "sparseness": "complexity",
    "model_parameter_randomisation": "randomisation",
    "random_logit": "randomisation",
}

# Metrics whose AR is inapplicable (they internally re-compute attributions).
# MPRT was disabled for the full ISIC run, so it has no reliability rows at
# all in the full-run CSV (pilot-era expectation included it for resnet18).
_NR_ONLY_RESNET18 = frozenset(
    {"avg_sensitivity", "max_sensitivity", "random_logit"}
)
_NR_ONLY_ALL_MODELS = frozenset({"avg_sensitivity", "max_sensitivity", "random_logit"})


@pytest.mark.skipif(
    not (_VERTICAL_SLICE.exists() and _RELIABILITY_CSV.exists()),
    reason="Both vertical_slice and meta_evaluation_reliability CSVs required for Part D",
)
class TestIntegrationMetaquantusDiscount:
    """Part D: discounted weights on real vertical-slice + reliability data."""

    def test_weights_sum_to_one_per_model(self):
        """Discounted weights must sum to 1.0 per model."""
        scores_df = pd.read_csv(_VERTICAL_SLICE)
        rel_df = pd.read_csv(_RELIABILITY_CSV)
        weights = metaquantus_discounted_weights(
            scores_df, rel_df, category_map=_QUANTUS_CATEGORIES
        )
        assert len(weights) > 0
        for group_key, w in weights.items():
            total = sum(w.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Weights don't sum to 1 for {group_key}: {total}"
            )

    def test_non_negative_weights(self):
        """All discounted weights must be >= 0."""
        scores_df = pd.read_csv(_VERTICAL_SLICE)
        rel_df = pd.read_csv(_RELIABILITY_CSV)
        weights = metaquantus_discounted_weights(
            scores_df, rel_df, category_map=_QUANTUS_CATEGORIES
        )
        for group_key, w in weights.items():
            for m, wt in w.items():
                assert wt >= 0.0, f"Negative weight for {group_key}/{m}: {wt}"

    def test_reliability_source_report(self, capsys):
        """Part D (c): report per-metric reliability source; assert NR-only set.

        Expected NR-only (AR inapplicable, metric re-computes attributions):
          - All models: avg_sensitivity, max_sensitivity, random_logit.
          - resnet18 only: model_parameter_randomisation (squeezenet nr_score
            aggregates to NaN across fae_methods → category fallback instead).
        """
        scores_df = pd.read_csv(_VERTICAL_SLICE)
        rel_df = pd.read_csv(_RELIABILITY_CSV)
        weights, sources = metaquantus_discounted_weights(
            scores_df, rel_df,
            category_map=_QUANTUS_CATEGORIES,
            return_sources=True,
        )

        # Collect sources keyed by (model, metric).
        per_model_metric: dict[tuple, str] = {}
        for group_key, src_map in sources.items():
            model = group_key[0]
            for m, src in src_map.items():
                per_model_metric[(model, m)] = src

        # --- Report (c) ---
        print("\n=== Part D (c): Per-metric reliability source ===")
        models = sorted({k[0] for k in per_model_metric})
        metrics = sorted({k[1] for k in per_model_metric})
        header = f"{'metric':<40}" + "".join(f"{m:<18}" for m in models)
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = f"{m:<40}"
            for model in models:
                row += f"{per_model_metric.get((model, m), 'N/A'):<18}"
            print(row)

        nr_only_per_model: dict[str, set] = {}
        for (model, m), src in per_model_metric.items():
            if src == "nr_only":
                nr_only_per_model.setdefault(model, set()).add(m)

        print("\nNR-only metrics per model:")
        for model in models:
            print(f"  {model}: {sorted(nr_only_per_model.get(model, set()))}")

        # --- Assertions ---
        # 3 metrics are NR-only for every model (AR universally inapplicable).
        for model in models:
            model_nr = nr_only_per_model.get(model, set())
            assert _NR_ONLY_ALL_MODELS <= model_nr, (
                f"Expected {_NR_ONLY_ALL_MODELS} ⊆ NR-only for {model}; got {model_nr}"
            )

        # resnet18: all 4 user-identified metrics are NR-only.
        resnet_nr = nr_only_per_model.get("resnet18", set())
        assert _NR_ONLY_RESNET18 <= resnet_nr, (
            f"Expected {_NR_ONLY_RESNET18} ⊆ NR-only for resnet18; got {resnet_nr}"
        )

        # squeezenet MPR: must NOT be NR-only (nr_score=NaN → category fallback).
        squeezenet_mpr_src = per_model_metric.get(("squeezenet", "model_parameter_randomisation"))
        assert squeezenet_mpr_src != "nr_only", (
            f"squeezenet MPR nr_score=NaN → expected category_fallback, got {squeezenet_mpr_src}"
        )
