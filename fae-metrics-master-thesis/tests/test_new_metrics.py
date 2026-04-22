"""Integration tests for the 9 new Quantus metrics in quantus_wrapper.py.

Uses a real ResNet-18 + real ISIC test image + real IG attribution to verify
each metric returns a finite float. Does NOT test numerical correctness
(Quantus has its own test suite for that).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aggregation.normalize import METRIC_DIRECTIONS
from src.attributions.generate import compute_integrated_gradients, compute_saliency
from src.data.isic_dataset import ISIC2017Dataset
from src.metrics.quantus_wrapper import _COMPLETENESS_METHODS, compute_all_metrics
from src.models.classifiers import load_resnet18

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Fixtures — loaded once per module for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return load_resnet18("weights/resnet18_isic2017.pth", device=DEVICE)


@pytest.fixture(scope="module")
def sample():
    ds = ISIC2017Dataset(root_dir="data", split="test", image_size=224, return_mask=True)
    return ds[0]


@pytest.fixture(scope="module")
def image_np(sample):
    return sample["image"].numpy()  # (3, 224, 224)


@pytest.fixture(scope="module")
def mask_np(sample):
    m = sample["mask"]
    return m.numpy() if m is not None else None  # (1, 224, 224)


@pytest.fixture(scope="module")
def target(model, sample):
    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0).to(DEVICE))
    return int(logits.argmax(dim=1).item())


@pytest.fixture(scope="module")
def ig_attribution(model, sample, target):
    attr = compute_integrated_gradients(model, sample["image"], target, device=DEVICE)
    return attr.numpy()


@pytest.fixture(scope="module")
def saliency_attribution(model, sample, target):
    attr = compute_saliency(model, sample["image"], target, device=DEVICE)
    return attr.numpy()


def _make_explain_func(model_ref):
    """Simple explain_func for metrics that need it."""
    def _explain(model, inputs, targets, **kwargs):
        attrs = []
        for i in range(inputs.shape[0]):
            img_t = torch.tensor(inputs[i], dtype=torch.float32)
            tgt = int(targets[i])
            a = compute_integrated_gradients(model_ref, img_t, tgt, device=DEVICE)
            attrs.append(a.numpy())
        return np.stack(attrs, axis=0)
    return _explain


@pytest.fixture(scope="module")
def explain_func(model):
    return _make_explain_func(model)


@pytest.fixture(scope="module")
def all_metric_scores(model, image_np, ig_attribution, target, mask_np, explain_func):
    return compute_all_metrics(
        model=model,
        image=image_np,
        attribution=ig_attribution,
        target=target,
        mask=mask_np,
        device=DEVICE,
        explain_func=explain_func,
        fae_method="integrated_gradients",
    )


# ---------------------------------------------------------------------------
# Test: each metric returns a finite float on IG attribution
# ---------------------------------------------------------------------------

_ALL_METRIC_NAMES = [
    "faithfulness_correlation",
    "pixel_flipping",
    "max_sensitivity",
    "avg_sensitivity",
    "relevance_mass_accuracy",
    "pointing_game",
    "sparseness",
    "complexity",
    "model_parameter_randomisation",
    "random_logit",
    "completeness",
    "non_sensitivity",
]


class TestAllMetricsFinite:
    @pytest.mark.parametrize("metric_name", _ALL_METRIC_NAMES)
    def test_metric_returns_finite_float(self, metric_name, all_metric_scores):
        """Each metric should return a finite float for IG (completeness-satisfying)."""
        val = all_metric_scores[metric_name]
        assert isinstance(val, float), f"{metric_name} should be float, got {type(val)}"
        assert np.isfinite(val), f"{metric_name} returned non-finite: {val}"

    def test_all_12_keys_present(self, all_metric_scores):
        assert len(all_metric_scores) == 12
        for name in _ALL_METRIC_NAMES:
            assert name in all_metric_scores, f"Missing key: {name}"


# ---------------------------------------------------------------------------
# Test: Completeness NaN for non-completeness methods
# ---------------------------------------------------------------------------

class TestCompletenessSkip:
    def test_completeness_nan_for_saliency(
        self, model, image_np, saliency_attribution, target, mask_np, explain_func,
    ):
        scores = compute_all_metrics(
            model=model,
            image=image_np,
            attribution=saliency_attribution,
            target=target,
            mask=mask_np,
            device=DEVICE,
            explain_func=explain_func,
            fae_method="saliency",
        )
        assert np.isnan(scores["completeness"]), (
            f"Completeness should be NaN for saliency, got {scores['completeness']}"
        )

    def test_completeness_nan_for_gradcam(
        self, model, image_np, saliency_attribution, target, mask_np, explain_func,
    ):
        """Use saliency attribution as a stand-in (shape is the same)."""
        scores = compute_all_metrics(
            model=model,
            image=image_np,
            attribution=saliency_attribution,
            target=target,
            mask=mask_np,
            device=DEVICE,
            explain_func=explain_func,
            fae_method="gradcam",
        )
        assert np.isnan(scores["completeness"])

    def test_completeness_finite_for_ig(self, all_metric_scores):
        """IG is a completeness-satisfying method — should return finite."""
        assert np.isfinite(all_metric_scores["completeness"])


# ---------------------------------------------------------------------------
# Test: METRIC_DIRECTIONS has entries for all 12 metrics
# ---------------------------------------------------------------------------

class TestMetricDirectionsRegistry:
    @pytest.mark.parametrize("metric_name", _ALL_METRIC_NAMES)
    def test_metric_in_directions_registry(self, metric_name):
        assert metric_name in METRIC_DIRECTIONS, (
            f"'{metric_name}' missing from METRIC_DIRECTIONS"
        )
        assert METRIC_DIRECTIONS[metric_name] in (+1, -1)

    def test_directions_count(self):
        """At least 12 entries in the registry."""
        assert len(METRIC_DIRECTIONS) >= 12
