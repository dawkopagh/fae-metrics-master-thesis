"""
Unified interface over the quantus library for metric computation.

Responsibilities (see docs/thesis_plan.md §6, Metric Computation):
    - Compute FaithfulnessCorrelation, MaxSensitivity, and
      RelevanceMassAccuracy for the vertical-slice scope
    - Return per-image metric scores as a dict[str, float]
    - Catch per-metric exceptions and return NaN with a logged warning

Not responsible for: redundancy analysis (see redundancy.py),
meta-validation (see meta_evaluation/), or aggregation (see aggregation/).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import quantus
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_three_metrics(
    model: nn.Module,
    image: np.ndarray,
    attribution: np.ndarray,
    mask: Optional[np.ndarray],
    target: int,
    device: str = "cpu",
    explain_func: Optional[Callable] = None,
    explain_func_kwargs: Optional[dict] = None,
) -> dict[str, float]:
    """Compute three evaluation metrics for a single image attribution.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    image : np.ndarray
        Input image of shape ``(3, H, W)`` (un-batched, channel-first).
    attribution : np.ndarray
        Attribution map of shape ``(3, H, W)`` (un-batched, channel-first).
    mask : np.ndarray or None
        Binary segmentation mask of shape ``(1, H, W)`` or ``None``.
        Used by RelevanceMassAccuracy.
    target : int
        Target class index.
    device : str
        Device for computation.
    explain_func : callable or None
        Callable with signature ``(model, inputs, targets, **kwargs) -> np.ndarray``
        that regenerates attributions on perturbed inputs. Required by
        MaxSensitivity.
    explain_func_kwargs : dict or None
        Additional keyword arguments forwarded to *explain_func*.

    Returns
    -------
    dict[str, float]
        Keys: ``'faithfulness_correlation'``, ``'max_sensitivity'``,
        ``'relevance_mass_accuracy'``. Values are floats or ``float('nan')``
        on failure.
    """
    # Quantus expects batched arrays: (1, C, H, W)
    x_batch = image[np.newaxis, ...]        # (1, 3, H, W)
    a_batch = attribution[np.newaxis, ...]  # (1, 3, H, W)
    y_batch = np.array([target])            # (1,)

    # Segmentation mask: Quantus expects s_batch shape (B, 1, H, W).
    s_batch: Optional[np.ndarray] = None
    if mask is not None:
        # mask shape from dataset: (1, H, W) → add batch dim → (1, 1, H, W)
        s_batch = mask[np.newaxis, ...]  # (1, 1, H, W)

    results: dict[str, float] = {}

    # --- FaithfulnessCorrelation ---
    try:
        fc = quantus.FaithfulnessCorrelation(
            nr_runs=100,
            subset_size=224,
            perturb_baseline="black",
            normalise=True,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = fc(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        )
        results["faithfulness_correlation"] = float(scores[0])
    except Exception as exc:
        logger.warning(
            "FaithfulnessCorrelation failed: %s: %s", type(exc).__name__, exc
        )
        results["faithfulness_correlation"] = float("nan")

    # --- MaxSensitivity ---
    try:
        ms = quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            normalise=False,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = ms(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs or {},
            device=device,
        )
        results["max_sensitivity"] = float(scores[0])
    except Exception as exc:
        logger.warning("MaxSensitivity failed: %s: %s", type(exc).__name__, exc)
        results["max_sensitivity"] = float("nan")

    # --- RelevanceMassAccuracy ---
    try:
        rma = quantus.RelevanceMassAccuracy(
            normalise=True,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        # RMA flattens both a_batch and s_batch to 2D and multiplies
        # element-wise. Since s_batch is (1, 1, H, W), reduce a_batch
        # from (1, C, H, W) to (1, 1, H, W) by summing over channels.
        a_batch_1ch = a_batch.sum(axis=1, keepdims=True)  # (1, 1, H, W)
        scores = rma(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch_1ch,
            s_batch=s_batch,
            channel_first=True,
            device=device,
        )
        results["relevance_mass_accuracy"] = float(scores[0])
    except Exception as exc:
        logger.warning(
            "RelevanceMassAccuracy failed: %s: %s", type(exc).__name__, exc
        )
        results["relevance_mass_accuracy"] = float("nan")

    return results
