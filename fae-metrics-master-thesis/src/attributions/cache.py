"""
Disk caching layer for pre-computed attribution maps.

Responsibilities (see docs/thesis_plan.md §6, Attribution Generation):
    - Save per-(model, FAE method, image_id) attribution maps to
      attributions_cache/ in a versioned, content-addressed format
    - Load cached attributions and return them in the canonical (3, H, W) shape
    - Invalidate cache entries when FAE hyperparameters change (via hash)
    - Avoid recomputing attributions that already exist on disk

Not responsible for: generating attributions from scratch (see generate.py),
ensembling attribution maps (see ensembling.py),
or metric computation (see metrics/).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _hyperparams_hash(hyperparams: dict) -> str:
    """Return a short (8-char) SHA-1 hex digest of sorted JSON hyperparams."""
    canonical = json.dumps(hyperparams, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:8]


class AttributionCache:
    """Disk cache for attribution tensors stored as ``.pt`` files.

    Directory layout::

        {root}/{model_arch}/{fae_method}/{image_id}_{target}_{hash}.pt

    Parameters
    ----------
    root : str
        Root directory for the cache. Default ``'attributions_cache'``.
    """

    def __init__(self, root: str = "attributions_cache") -> None:
        self.root = Path(root)

    def key(
        self,
        model_arch: str,
        fae_method: str,
        image_id: str,
        target: int,
        fae_hyperparams: dict,
    ) -> str:
        """Build the relative cache path (without ``.pt`` suffix).

        Parameters
        ----------
        model_arch : str
            Architecture name, e.g. ``'resnet18'``.
        fae_method : str
            FAE method name, e.g. ``'integrated_gradients'``.
        image_id : str
            Image identifier, e.g. ``'ISIC_0012258'``.
        target : int
            Target class index used for attribution.
        fae_hyperparams : dict
            Method hyperparameters whose hash guards against stale entries.

        Returns
        -------
        str
            Relative path like
            ``resnet18/integrated_gradients/ISIC_0012258_0_a3b1c2d4``.
        """
        h = _hyperparams_hash(fae_hyperparams)
        return f"{model_arch}/{fae_method}/{image_id}_{target}_{h}"

    def _path(
        self,
        model_arch: str,
        fae_method: str,
        image_id: str,
        target: int,
        fae_hyperparams: dict,
    ) -> Path:
        """Return the absolute ``.pt`` file path for the given key."""
        k = self.key(model_arch, fae_method, image_id, target, fae_hyperparams)
        return self.root / f"{k}.pt"

    def exists(
        self,
        model_arch: str,
        fae_method: str,
        image_id: str,
        target: int,
        fae_hyperparams: dict,
    ) -> bool:
        """Check whether a cached attribution exists on disk."""
        return self._path(model_arch, fae_method, image_id, target, fae_hyperparams).is_file()

    def get(
        self,
        model_arch: str,
        fae_method: str,
        image_id: str,
        target: int,
        fae_hyperparams: dict,
    ) -> torch.Tensor | None:
        """Load a cached attribution tensor, or return ``None`` on miss.

        Returns
        -------
        torch.Tensor or None
            Attribution tensor (typically shape ``(3, H, W)``) or ``None``.
        """
        p = self._path(model_arch, fae_method, image_id, target, fae_hyperparams)
        if not p.is_file():
            return None
        logger.debug("Cache hit: %s", p)
        return torch.load(p, map_location="cpu", weights_only=True)

    def put(
        self,
        model_arch: str,
        fae_method: str,
        image_id: str,
        target: int,
        fae_hyperparams: dict,
        attribution: torch.Tensor,
    ) -> None:
        """Save an attribution tensor to the cache.

        Parameters
        ----------
        attribution : torch.Tensor
            Attribution map, typically shape ``(3, H, W)``.
        """
        p = self._path(model_arch, fae_method, image_id, target, fae_hyperparams)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(attribution, p)
        logger.debug("Cached: %s", p)
