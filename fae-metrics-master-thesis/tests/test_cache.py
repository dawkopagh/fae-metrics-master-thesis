"""Tests for src/attributions/cache — AttributionCache and compute_or_load."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.attributions.cache import AttributionCache, _hyperparams_hash
from src.attributions.generate import compute_or_load


@pytest.fixture()
def cache(tmp_path: Path) -> AttributionCache:
    """Return a fresh AttributionCache rooted in a temp directory."""
    return AttributionCache(root=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# AttributionCache.key
# ---------------------------------------------------------------------------

class TestKey:
    def test_key_format(self, cache: AttributionCache) -> None:
        k = cache.key("resnet18", "saliency", "ISIC_0012258", 0, {})
        parts = k.split("/")
        assert parts[0] == "resnet18"
        assert parts[1] == "saliency"
        assert parts[2].startswith("ISIC_0012258_0_")

    def test_key_deterministic(self, cache: AttributionCache) -> None:
        hp = {"n_steps": 50}
        k1 = cache.key("resnet18", "ig", "IMG_001", 1, hp)
        k2 = cache.key("resnet18", "ig", "IMG_001", 1, hp)
        assert k1 == k2

    def test_key_changes_with_hyperparams(self, cache: AttributionCache) -> None:
        k1 = cache.key("resnet18", "ig", "IMG_001", 0, {"n_steps": 50})
        k2 = cache.key("resnet18", "ig", "IMG_001", 0, {"n_steps": 100})
        assert k1 != k2


# ---------------------------------------------------------------------------
# put / get
# ---------------------------------------------------------------------------

class TestPutGet:
    def test_put_then_get_returns_equal_tensor(self, cache: AttributionCache) -> None:
        attr = torch.randn(3, 224, 224)
        hp = {"n_steps": 50}
        cache.put("resnet18", "ig", "ISIC_001", 0, hp, attr)
        loaded = cache.get("resnet18", "ig", "ISIC_001", 0, hp)
        assert loaded is not None
        assert torch.equal(attr, loaded)

    def test_get_miss_returns_none(self, cache: AttributionCache) -> None:
        result = cache.get("resnet18", "ig", "ISIC_999", 0, {})
        assert result is None

    def test_exists_after_put(self, cache: AttributionCache) -> None:
        attr = torch.randn(3, 8, 8)
        hp = {"k": 5}
        assert not cache.exists("sq", "sal", "IMG_X", 2, hp)
        cache.put("sq", "sal", "IMG_X", 2, hp, attr)
        assert cache.exists("sq", "sal", "IMG_X", 2, hp)


# ---------------------------------------------------------------------------
# Hyperparameter-change invalidation
# ---------------------------------------------------------------------------

class TestHyperparamInvalidation:
    def test_different_hyperparams_miss(self, cache: AttributionCache) -> None:
        attr = torch.randn(3, 8, 8)
        cache.put("resnet18", "ig", "IMG_001", 0, {"n_steps": 50}, attr)
        # Same everything except n_steps → should miss
        loaded = cache.get("resnet18", "ig", "IMG_001", 0, {"n_steps": 100})
        assert loaded is None

    def test_different_target_miss(self, cache: AttributionCache) -> None:
        attr = torch.randn(3, 8, 8)
        cache.put("resnet18", "ig", "IMG_001", 0, {}, attr)
        loaded = cache.get("resnet18", "ig", "IMG_001", 1, {})
        assert loaded is None


# ---------------------------------------------------------------------------
# _hyperparams_hash
# ---------------------------------------------------------------------------

class TestHyperparamsHash:
    def test_order_invariant(self) -> None:
        h1 = _hyperparams_hash({"a": 1, "b": 2})
        h2 = _hyperparams_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_empty_dict(self) -> None:
        h = _hyperparams_hash({})
        assert len(h) == 8


# ---------------------------------------------------------------------------
# compute_or_load
# ---------------------------------------------------------------------------

class TestComputeOrLoad:
    def test_computes_on_miss_and_caches(self, cache: AttributionCache) -> None:
        expected = torch.randn(3, 8, 8)
        call_count = 0

        def fake_compute(**kwargs) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return expected

        result = compute_or_load(
            cache=cache,
            compute_fn=fake_compute,
            model_arch="resnet18",
            fae_method="ig",
            image_id="IMG_001",
            target=0,
            fae_hyperparams={"n_steps": 50},
        )
        assert call_count == 1
        assert torch.equal(result, expected)
        # Second call should hit cache
        result2 = compute_or_load(
            cache=cache,
            compute_fn=fake_compute,
            model_arch="resnet18",
            fae_method="ig",
            image_id="IMG_001",
            target=0,
            fae_hyperparams={"n_steps": 50},
        )
        assert call_count == 1  # not called again
        assert torch.equal(result2, expected)

    def test_bypasses_cache_when_none(self) -> None:
        call_count = 0

        def fake_compute(**kwargs) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return torch.zeros(3, 8, 8)

        compute_or_load(
            cache=None,
            compute_fn=fake_compute,
            model_arch="resnet18",
            fae_method="ig",
            image_id="IMG_001",
            target=0,
            fae_hyperparams={},
        )
        compute_or_load(
            cache=None,
            compute_fn=fake_compute,
            model_arch="resnet18",
            fae_method="ig",
            image_id="IMG_001",
            target=0,
            fae_hyperparams={},
        )
        assert call_count == 2  # always recomputed
