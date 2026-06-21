"""
NormEnsembleXAI wrapper: ensemble multiple attribution maps into a single map.

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 3;
and §10, Decision D6):
    - Implement the NormEnsembleXAI Mean aggregation with Second Moment
      Scaling normalisation (Hryniewska-Guzik et al., 2024) as the
      default ensembling strategy
    - Accept a dict of {fae_method: attribution_map} and return one
      ensembled map in the canonical (3, H, W) shape produced by
      generate.py, so it can be fed straight into the Quantus metrics
    - Support choosing the subset of methods to ensemble
    - Support alternative aggregation operators (sum, max) for ablation

Not responsible for: generating individual attributions (see generate.py),
metric computation on ensembled maps (see metrics/),
or the multi-criteria aggregation of metric scores (see aggregation/).
"""

from __future__ import annotations

import logging
from typing import Literal, Mapping, Sequence

import numpy as np
import torch

# Reuse the Second Moment Scaling implementation rather than reimplementing
# it.  ``second_moment_scale`` divides a 1-D array by its root-mean-square
# (s / sqrt(E[s^2])); applied to a flattened attribution map it rescales the
# whole map to unit second moment while preserving sign structure.  Per
# Decision D6 the attribution-level direction is always +1 (we are not
# rectifying a quality metric here, only normalising magnitude).
from src.aggregation.normalize import second_moment_scale

logger = logging.getLogger(__name__)

Aggregation = Literal["mean", "sum", "max"]

# Default per-map normalisation direction.  Attribution maps carry signed
# relevance; ensembling normalises magnitude only, so direction is fixed at
# +1 (no sign flip) — unlike the metric-score normalisation in normalize.py.
_ATTR_DIRECTION = 1


def second_moment_normalize_map(attribution: torch.Tensor) -> torch.Tensor:
    """Apply Second Moment Scaling to a single attribution map.

    The map is flattened, rescaled by :func:`second_moment_scale`
    (``a / sqrt(E[a^2])``), and reshaped back to its original shape.
    This preserves the sign and spatial structure of *attribution* while
    mapping its "typical" magnitude to ±1, putting heterogeneous FAE
    methods (e.g. unbounded IG vs. non-negative Saliency) on a common
    scale before aggregation.

    Parameters
    ----------
    attribution : torch.Tensor
        Attribution map of any shape (canonical ``(3, H, W)``), real-valued.

    Returns
    -------
    torch.Tensor
        Normalised attribution map with the same shape, dtype, and device
        as *attribution*.  An all-zero map is returned unchanged (zeros),
        matching :func:`second_moment_scale`.
    """
    flat = attribution.detach().cpu().numpy().reshape(-1)
    normed = second_moment_scale(flat, direction=_ATTR_DIRECTION)
    out = torch.from_numpy(
        normed.reshape(tuple(attribution.shape))
    ).to(dtype=attribution.dtype, device=attribution.device)
    return out


def _aggregate(stacked: torch.Tensor, aggregation: Aggregation) -> torch.Tensor:
    """Reduce a ``(N, ...)`` stack of normalised maps to a single map.

    Parameters
    ----------
    stacked : torch.Tensor
        Stack of *N* normalised maps along dim 0.
    aggregation : {'mean', 'sum', 'max'}
        Reduction operator. ``'mean'`` is the Decision-D6 default.

    Returns
    -------
    torch.Tensor
        Aggregated map of shape ``stacked.shape[1:]``.
    """
    if aggregation == "mean":
        return stacked.mean(dim=0)
    if aggregation == "sum":
        return stacked.sum(dim=0)
    if aggregation == "max":
        return stacked.max(dim=0).values
    raise ValueError(
        f"Unknown aggregation '{aggregation}'. Choose from 'mean', 'sum', 'max'."
    )


def ensemble_attributions(
    attributions: Mapping[str, torch.Tensor],
    methods: Sequence[str] | None = None,
    aggregation: Aggregation = "mean",
) -> torch.Tensor:
    """Ensemble per-method attribution maps into one map (NormEnsembleXAI).

    Implements Decision D6: each selected method's map is first normalised
    with Second Moment Scaling (:func:`second_moment_normalize_map`) and the
    normalised maps are then combined with *aggregation* (default ``'mean'``).
    Normalisation before aggregation prevents a single high-magnitude method
    from dominating the ensemble.

    Reference: Hryniewska-Guzik et al. (2024), NormEnsembleXAI.

    Parameters
    ----------
    attributions : Mapping[str, torch.Tensor]
        Per-method attribution maps for the SAME ``(model, image, target)``.
        Keys are FAE method names (see :data:`generate.FAE_METHODS`); values
        are tensors in the canonical ``(3, H, W)`` shape from generate.py.
        All selected maps must share the same shape.
    methods : Sequence[str] or None
        Subset of method names to ensemble, in any order.  If ``None``, all
        keys of *attributions* are used (sorted for determinism).  Every name
        must be present in *attributions*.
    aggregation : {'mean', 'sum', 'max'}
        Aggregation operator applied across the normalised maps.  Default
        ``'mean'`` per Decision D6; ``'sum'``/``'max'`` are for ablation.

    Returns
    -------
    torch.Tensor
        The ensembled attribution map, same shape/dtype/device as the inputs,
        ready to pass to :func:`metrics.quantus_wrapper.compute_all_metrics`
        (after ``.numpy()`` as the pipeline already does for single methods).

    Raises
    ------
    ValueError
        If *methods* (or *attributions*) is empty, a requested method is
        missing, or the selected maps have mismatched shapes.
    KeyError
        Propagated from selecting a method absent in *attributions* — surfaced
        as a ``ValueError`` with the offending name for clarity.
    """
    if methods is None:
        selected = sorted(attributions.keys())
    else:
        selected = list(methods)

    if len(selected) == 0:
        raise ValueError(
            "No methods to ensemble: 'attributions' is empty or 'methods' is []."
        )

    missing = [m for m in selected if m not in attributions]
    if missing:
        raise ValueError(
            f"Method(s) {missing} not found in attributions "
            f"(available: {sorted(attributions.keys())})."
        )

    ref_shape = tuple(attributions[selected[0]].shape)
    for m in selected:
        shp = tuple(attributions[m].shape)
        if shp != ref_shape:
            raise ValueError(
                f"Attribution shape mismatch: '{m}' has shape {shp}, "
                f"expected {ref_shape} (from '{selected[0]}')."
            )

    if len(selected) == 1:
        # Single-method "ensemble": normalise and return the lone map.
        logger.debug("Single-method ensemble for '%s'.", selected[0])
        return second_moment_normalize_map(attributions[selected[0]])

    logger.debug(
        "Ensembling %d methods %s with '%s' aggregation.",
        len(selected), selected, aggregation,
    )

    normed = [second_moment_normalize_map(attributions[m]) for m in selected]
    stacked = torch.stack(normed, dim=0)  # (N, 3, H, W)
    return _aggregate(stacked, aggregation)


def ensemble_attributions_numpy(
    attributions: Mapping[str, np.ndarray | torch.Tensor],
    methods: Sequence[str] | None = None,
    aggregation: Aggregation = "mean",
) -> np.ndarray:
    """Convenience wrapper returning a NumPy ensembled map.

    Identical to :func:`ensemble_attributions` but accepts NumPy arrays or
    tensors and returns a ``np.ndarray``.  This matches the array contract
    of :func:`metrics.quantus_wrapper.compute_all_metrics`, which expects a
    channel-first ``(3, H, W)`` NumPy attribution.

    Parameters
    ----------
    attributions : Mapping[str, np.ndarray or torch.Tensor]
        Per-method maps for one ``(model, image, target)``.
    methods : Sequence[str] or None
        Subset of methods to ensemble (see :func:`ensemble_attributions`).
    aggregation : {'mean', 'sum', 'max'}
        Aggregation operator.

    Returns
    -------
    np.ndarray
        Ensembled attribution as a ``float64`` channel-first array.
    """
    as_tensor: dict[str, torch.Tensor] = {}
    for name, attr in attributions.items():
        if isinstance(attr, torch.Tensor):
            as_tensor[name] = attr
        else:
            as_tensor[name] = torch.as_tensor(np.asarray(attr))
    ensembled = ensemble_attributions(
        as_tensor, methods=methods, aggregation=aggregation
    )
    return ensembled.detach().cpu().numpy()
