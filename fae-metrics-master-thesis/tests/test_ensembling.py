"""Tests for src/attributions/ensembling.py (NormEnsembleXAI, Decision D6).

All fixtures are deterministic synthetic tensors — no model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aggregation.normalize import second_moment_scale
from src.attributions.ensembling import (
    ensemble_attributions,
    ensemble_attributions_numpy,
    second_moment_normalize_map,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_map(seed: int, shape=(3, 8, 8)) -> torch.Tensor:
    """Deterministic signed float32 attribution map of canonical shape."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)


# ── second_moment_normalize_map ───────────────────────────────────────────

class TestSecondMomentNormalizeMap:
    def test_matches_flattened_second_moment_scale(self):
        """Map normalisation == flatten → second_moment_scale → reshape."""
        attr = _make_map(0)
        out = second_moment_normalize_map(attr)
        expected = second_moment_scale(
            attr.numpy().reshape(-1), direction=1
        ).reshape(attr.shape)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_unit_second_moment(self):
        """After normalisation, sqrt(mean(a^2)) == 1 for a non-zero map."""
        attr = _make_map(1)
        out = second_moment_normalize_map(attr)
        rms = float(np.sqrt(np.mean(out.numpy() ** 2)))
        assert rms == pytest.approx(1.0, rel=1e-5)

    def test_all_zero_map_returns_zeros(self):
        attr = torch.zeros(3, 4, 4)
        out = second_moment_normalize_map(attr)
        assert torch.equal(out, torch.zeros(3, 4, 4))

    def test_shape_and_dtype_preserved(self):
        attr = _make_map(2, shape=(3, 6, 6))
        out = second_moment_normalize_map(attr)
        assert out.shape == attr.shape
        assert out.dtype == attr.dtype


# ── ensemble_attributions ─────────────────────────────────────────────────

class TestEnsembleAttributions:
    def test_identical_maps_return_normalized_map(self):
        """(a) Ensembling identical maps == the normalized single map.

        mean of N copies of the same normalised map is that map.
        """
        attr = _make_map(3)
        maps = {f"m{i}": attr.clone() for i in range(4)}
        out = ensemble_attributions(maps, aggregation="mean")
        expected = second_moment_normalize_map(attr)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_equals_normalized_mean(self):
        """(b) Ensemble of N maps == mean of their per-map normalisations."""
        maps = {f"m{i}": _make_map(10 + i) for i in range(3)}
        out = ensemble_attributions(maps, aggregation="mean")

        normed = [second_moment_normalize_map(maps[k]) for k in sorted(maps)]
        expected = torch.stack(normed, dim=0).mean(dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_shape_dtype_preserved(self):
        """(c) Output keeps canonical (3, H, W) shape and dtype."""
        maps = {f"m{i}": _make_map(20 + i, shape=(3, 5, 5)) for i in range(3)}
        out = ensemble_attributions(maps)
        assert out.shape == (3, 5, 5)
        assert out.dtype == torch.float32

    def test_single_method_ensemble(self):
        """(d) Single-method ensemble returns the normalized lone map."""
        attr = _make_map(30)
        out_dict = ensemble_attributions({"only": attr})
        out_subset = ensemble_attributions(
            {"only": attr, "ignored": _make_map(31)}, methods=["only"]
        )
        expected = second_moment_normalize_map(attr)
        np.testing.assert_allclose(out_dict.numpy(), expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(out_subset.numpy(), expected.numpy(), rtol=1e-6)

    def test_method_subset_selection(self):
        """Only the selected subset participates in the ensemble."""
        maps = {"a": _make_map(40), "b": _make_map(41), "c": _make_map(42)}
        out = ensemble_attributions(maps, methods=["a", "c"])
        normed = [second_moment_normalize_map(maps[k]) for k in ["a", "c"]]
        expected = torch.stack(normed, dim=0).mean(dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_default_methods_sorted_deterministic(self):
        """methods=None uses all keys in a deterministic (sorted) order."""
        maps = {"z": _make_map(50), "a": _make_map(51)}
        out1 = ensemble_attributions(maps)
        out2 = ensemble_attributions(maps, methods=sorted(maps))
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-12)

    def test_sum_aggregation(self):
        maps = {f"m{i}": _make_map(60 + i) for i in range(2)}
        out = ensemble_attributions(maps, aggregation="sum")
        normed = [second_moment_normalize_map(maps[k]) for k in sorted(maps)]
        expected = torch.stack(normed, dim=0).sum(dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_max_aggregation(self):
        maps = {f"m{i}": _make_map(70 + i) for i in range(2)}
        out = ensemble_attributions(maps, aggregation="max")
        normed = [second_moment_normalize_map(maps[k]) for k in sorted(maps)]
        expected = torch.stack(normed, dim=0).max(dim=0).values
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    # ── error handling ────────────────────────────────────────────────────

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No methods"):
            ensemble_attributions({})

    def test_empty_methods_list_raises(self):
        with pytest.raises(ValueError, match="No methods"):
            ensemble_attributions({"a": _make_map(80)}, methods=[])

    def test_missing_method_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ensemble_attributions({"a": _make_map(81)}, methods=["a", "b"])

    def test_shape_mismatch_raises(self):
        maps = {"a": _make_map(82, (3, 8, 8)), "b": _make_map(83, (3, 4, 4))}
        with pytest.raises(ValueError, match="shape mismatch"):
            ensemble_attributions(maps)

    def test_unknown_aggregation_raises(self):
        maps = {f"m{i}": _make_map(90 + i) for i in range(2)}
        with pytest.raises(ValueError, match="Unknown aggregation"):
            ensemble_attributions(maps, aggregation="median")  # type: ignore[arg-type]


# ── ensemble_attributions_numpy ───────────────────────────────────────────

class TestEnsembleAttributionsNumpy:
    def test_numpy_matches_tensor_path(self):
        maps_t = {f"m{i}": _make_map(100 + i) for i in range(3)}
        maps_np = {k: v.numpy() for k, v in maps_t.items()}
        out_np = ensemble_attributions_numpy(maps_np)
        out_t = ensemble_attributions(maps_t).numpy()
        assert isinstance(out_np, np.ndarray)
        np.testing.assert_allclose(out_np, out_t, rtol=1e-6)

    def test_numpy_output_shape(self):
        maps = {f"m{i}": _make_map(110 + i, (3, 7, 7)).numpy() for i in range(2)}
        out = ensemble_attributions_numpy(maps)
        assert out.shape == (3, 7, 7)
