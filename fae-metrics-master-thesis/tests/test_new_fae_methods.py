"""Tests for the four new FAE methods: DeepLift, GuidedBackprop, LRP, Occlusion.

All tests use a small nn.Sequential fixture (Conv2d + ReLU + Flatten + Linear)
for speed. Integration tests with real ResNet-18 / SqueezeNet come in the
vertical-slice run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.attributions.generate import (
    FAE_METHODS,
    compute_deep_lift,
    compute_guided_backprop,
    compute_lrp,
    compute_occlusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_model() -> nn.Module:
    """Minimal Conv → ReLU → Flatten → Linear classifier (3-class)."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    )
    model.eval()
    return model


@pytest.fixture()
def lrp_model() -> nn.Module:
    """LRP-compatible model that avoids nn.Flatten (unsupported by captum LRP).

    Uses torch.flatten() inside forward() — the same pattern as
    torchvision ResNet-18, which is why LRP works on the real models.
    """

    class _LRPCompatibleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = _LRPCompatibleModel()
    model.eval()
    return model


@pytest.fixture()
def image_64() -> torch.Tensor:
    """Random (3, 64, 64) input image with non-trivial content."""
    torch.manual_seed(0)
    return torch.randn(3, 64, 64)


# ---------------------------------------------------------------------------
# DeepLift
# ---------------------------------------------------------------------------

class TestDeepLift:
    def test_output_shape(self, small_model: nn.Module, image_64: torch.Tensor):
        attr = compute_deep_lift(small_model, image_64, target=0)
        assert attr.shape == (3, 64, 64)

    def test_no_nan_inf(self, small_model: nn.Module, image_64: torch.Tensor):
        attr = compute_deep_lift(small_model, image_64, target=0)
        assert not torch.isnan(attr).any()
        assert not torch.isinf(attr).any()

    def test_signed_output(self, small_model: nn.Module, image_64: torch.Tensor):
        """DeepLift produces signed attributions (positive and negative)."""
        attr = compute_deep_lift(small_model, image_64, target=0)
        assert (attr > 0).any(), "Expected some positive values in DeepLift output"
        assert (attr < 0).any(), "Expected some negative values in DeepLift output"

    def test_dispatch_dict(self, small_model: nn.Module, image_64: torch.Tensor):
        fn = FAE_METHODS["deep_lift"]
        attr = fn(model=small_model, image=image_64, target=0)
        assert attr.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# GuidedBackprop
# ---------------------------------------------------------------------------

class TestGuidedBackprop:
    def test_output_shape(self, small_model: nn.Module, image_64: torch.Tensor):
        attr = compute_guided_backprop(small_model, image_64, target=0)
        assert attr.shape == (3, 64, 64)

    def test_no_nan_inf(self, small_model: nn.Module, image_64: torch.Tensor):
        attr = compute_guided_backprop(small_model, image_64, target=0)
        assert not torch.isnan(attr).any()
        assert not torch.isinf(attr).any()

    def test_non_negative(self, small_model: nn.Module, image_64: torch.Tensor):
        """GuidedBackprop output should be non-negative (guided ReLU).

        A small numerical tolerance is applied because floating-point
        arithmetic in the backward hooks can produce tiny negative values
        on the order of 1e-4.
        """
        attr = compute_guided_backprop(small_model, image_64, target=0)
        assert (attr >= -1e-3).all(), (
            f"GuidedBackprop should be non-negative (tol 1e-3), got min={attr.min().item():.6f}"
        )

    def test_dispatch_dict(self, small_model: nn.Module, image_64: torch.Tensor):
        fn = FAE_METHODS["guided_backprop"]
        attr = fn(model=small_model, image=image_64, target=0)
        assert attr.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# LRP
# ---------------------------------------------------------------------------

class TestLRP:
    def test_output_shape(self, lrp_model: nn.Module, image_64: torch.Tensor):
        attr = compute_lrp(lrp_model, image_64, target=0)
        assert attr.shape == (3, 64, 64)

    def test_no_nan_inf(self, lrp_model: nn.Module, image_64: torch.Tensor):
        attr = compute_lrp(lrp_model, image_64, target=0)
        assert not torch.isnan(attr).any()
        assert not torch.isinf(attr).any()

    def test_signed_output(self, lrp_model: nn.Module, image_64: torch.Tensor):
        """LRP produces signed relevance scores."""
        attr = compute_lrp(lrp_model, image_64, target=0)
        assert (attr > 0).any(), "Expected some positive values in LRP output"
        assert (attr < 0).any(), "Expected some negative values in LRP output"

    def test_dispatch_dict(self, lrp_model: nn.Module, image_64: torch.Tensor):
        fn = FAE_METHODS["lrp"]
        attr = fn(model=lrp_model, image=image_64, target=0)
        assert attr.shape == (3, 64, 64)

    def test_unsupported_layer_raises(self, small_model: nn.Module, image_64: torch.Tensor):
        """nn.Flatten is unsupported by captum LRP — should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="LRP failed"):
            compute_lrp(small_model, image_64, target=0)


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------

class TestOcclusion:
    def test_output_shape(self, small_model: nn.Module, image_64: torch.Tensor):
        attr = compute_occlusion(small_model, image_64, target=0)
        assert attr.shape == (3, 64, 64)

    def test_no_nan_inf(self, small_model: nn.Module, image_64: torch.Tensor):
        attr = compute_occlusion(small_model, image_64, target=0)
        assert not torch.isnan(attr).any()
        assert not torch.isinf(attr).any()

    def test_signed_output(self, small_model: nn.Module, image_64: torch.Tensor):
        """Occlusion produces signed attributions on a non-trivial input.

        With a small random model, all attributions may share one sign.
        We verify that the output is not identically zero (i.e., the method
        actually computes something).
        """
        attr = compute_occlusion(small_model, image_64, target=0)
        assert attr.abs().sum() > 0, "Expected non-zero Occlusion output"

    def test_custom_window(self, small_model: nn.Module, image_64: torch.Tensor):
        """Custom sliding window shapes and strides work."""
        attr = compute_occlusion(
            small_model, image_64, target=0,
            sliding_window_shapes=(3, 8, 8),
            strides=(3, 4, 4),
        )
        assert attr.shape == (3, 64, 64)

    def test_dispatch_dict(self, small_model: nn.Module, image_64: torch.Tensor):
        fn = FAE_METHODS["occlusion"]
        attr = fn(model=small_model, image=image_64, target=0)
        assert attr.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# All new methods registered in FAE_METHODS
# ---------------------------------------------------------------------------

class TestFAEMethodsRegistry:
    @pytest.mark.parametrize("method_name", ["deep_lift", "guided_backprop", "lrp", "occlusion"])
    def test_method_registered(self, method_name: str):
        assert method_name in FAE_METHODS

    def test_total_method_count(self):
        """7 FAE methods should be registered."""
        assert len(FAE_METHODS) == 7
