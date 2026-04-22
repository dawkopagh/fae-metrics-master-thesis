"""
Unified interface over the quantus library for metric computation.

Responsibilities (see docs/thesis_plan.md §6, Metric Computation):
    - Compute 12 evaluation metrics across 6 Quantus categories:
        Faithfulness:    FaithfulnessCorrelation (+1), PixelFlipping (+1)
        Robustness:      MaxSensitivity (-1), AvgSensitivity (-1)
        Localization:    RelevanceMassAccuracy (+1), PointingGame (+1)
        Complexity:      Sparseness (+1), Complexity (-1)
        Randomization:   ModelParameterRandomisation (-1), RandomLogit (-1)
        Axiomatic:       Completeness (-1), NonSensitivity (-1)
    - Return per-image metric scores as a dict[str, float]
    - Catch per-metric exceptions and return NaN with a logged warning
    - Skip Completeness for FAE methods that do not satisfy the axiom

Direction conventions (higher_is_better = +1, lower_is_better = -1):
    These MUST match METRIC_DIRECTIONS in src/aggregation/normalize.py.

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

# FAE methods for which the Completeness axiom is meaningful.
# Running Completeness on methods outside this set returns NaN.
_COMPLETENESS_METHODS = frozenset({
    "integrated_gradients",
    "deep_lift",
    "lrp",
})


def compute_all_metrics(
    model: nn.Module,
    image: np.ndarray,
    attribution: np.ndarray,
    target: int,
    mask: Optional[np.ndarray] = None,
    device: str = "cpu",
    explain_func: Optional[Callable] = None,
    explain_func_kwargs: Optional[dict] = None,
    fae_method: Optional[str] = None,
    num_classes: int = 3,
    include_non_sensitivity: bool = False,
) -> dict[str, float]:
    """Compute evaluation metrics for a single image attribution.

    By default computes 11 metrics. NonSensitivity is excluded because
    it requires ~9k forward passes per image (Quantus features_in_step
    bug forces features_in_step=1). Set ``include_non_sensitivity=True``
    to include it at the cost of significant runtime.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    image : np.ndarray
        Input image of shape ``(3, H, W)`` (un-batched, channel-first).
    attribution : np.ndarray
        Attribution map of shape ``(3, H, W)`` (un-batched, channel-first).
    target : int
        Target class index.
    mask : np.ndarray or None
        Binary segmentation mask of shape ``(1, H, W)`` or ``None``.
        Used by RelevanceMassAccuracy and PointingGame.
    device : str
        Device for computation.
    explain_func : callable or None
        Callable with signature ``(model, inputs, targets, **kwargs) -> np.ndarray``
        that regenerates attributions on perturbed inputs. Required by
        MaxSensitivity, AvgSensitivity, ModelParameterRandomisation.
    explain_func_kwargs : dict or None
        Additional keyword arguments forwarded to *explain_func*.
    fae_method : str or None
        Name of the FAE method that produced *attribution*. Used to
        decide whether Completeness is applicable.
    num_classes : int
        Number of output classes (for RandomLogit). Default 3 (ISIC 2017).

    Returns
    -------
    dict[str, float]
        Keys are metric snake_case names. Values are floats or
        ``float('nan')`` on failure or intentional skip (Completeness
        on non-completeness methods).
    """
    # Quantus expects batched arrays: (1, C, H, W)
    x_batch = image[np.newaxis, ...]        # (1, 3, H, W)
    a_batch = attribution[np.newaxis, ...]  # (1, 3, H, W)
    y_batch = np.array([target])            # (1,)

    # Segmentation mask: Quantus expects s_batch shape (B, 1, H, W).
    s_batch: Optional[np.ndarray] = None
    if mask is not None:
        s_batch = mask[np.newaxis, ...]  # (1, 1, H, W)

    # Single-channel attribution for localization metrics
    a_batch_1ch = a_batch.sum(axis=1, keepdims=True)  # (1, 1, H, W)

    ef = explain_func
    ef_kwargs = explain_func_kwargs or {}

    results: dict[str, float] = {}

    # =====================================================================
    # FAITHFULNESS
    # =====================================================================

    # --- FaithfulnessCorrelation (direction: +1) ---
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
        logger.warning("FaithfulnessCorrelation failed: %s: %s", type(exc).__name__, exc)
        results["faithfulness_correlation"] = float("nan")

    # --- PixelFlipping (direction: +1, AUC — higher means more faithful) ---
    try:
        pf = quantus.PixelFlipping(
            features_in_step=224,
            perturb_baseline="black",
            normalise=True,
            abs=False,
            return_aggregate=False,
            return_auc_per_sample=True,
            disable_warnings=True,
        )
        scores = pf(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        )
        results["pixel_flipping"] = float(scores[0])
    except Exception as exc:
        logger.warning("PixelFlipping failed: %s: %s", type(exc).__name__, exc)
        results["pixel_flipping"] = float("nan")

    # =====================================================================
    # ROBUSTNESS
    # =====================================================================

    # --- MaxSensitivity (direction: -1) ---
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
            explain_func=ef,
            explain_func_kwargs=ef_kwargs,
            device=device,
        )
        results["max_sensitivity"] = float(scores[0])
    except Exception as exc:
        logger.warning("MaxSensitivity failed: %s: %s", type(exc).__name__, exc)
        results["max_sensitivity"] = float("nan")

    # --- AvgSensitivity (direction: -1) ---
    try:
        avgs = quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            normalise=False,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = avgs(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            explain_func=ef,
            explain_func_kwargs=ef_kwargs,
            device=device,
        )
        results["avg_sensitivity"] = float(scores[0])
    except Exception as exc:
        logger.warning("AvgSensitivity failed: %s: %s", type(exc).__name__, exc)
        results["avg_sensitivity"] = float("nan")

    # =====================================================================
    # LOCALIZATION
    # =====================================================================

    # --- RelevanceMassAccuracy (direction: +1) ---
    try:
        rma = quantus.RelevanceMassAccuracy(
            normalise=True,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
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
        logger.warning("RelevanceMassAccuracy failed: %s: %s", type(exc).__name__, exc)
        results["relevance_mass_accuracy"] = float("nan")

    # --- PointingGame (direction: +1) ---
    try:
        pg = quantus.PointingGame(
            normalise=True,
            abs=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = pg(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch_1ch,
            s_batch=s_batch,
            channel_first=True,
            device=device,
        )
        results["pointing_game"] = float(scores[0])
    except Exception as exc:
        logger.warning("PointingGame failed: %s: %s", type(exc).__name__, exc)
        results["pointing_game"] = float("nan")

    # =====================================================================
    # COMPLEXITY
    # =====================================================================

    # --- Sparseness (direction: +1, Gini-like) ---
    try:
        sp = quantus.Sparseness(
            abs=True,
            normalise=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = sp(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        )
        results["sparseness"] = float(scores[0])
    except Exception as exc:
        logger.warning("Sparseness failed: %s: %s", type(exc).__name__, exc)
        results["sparseness"] = float("nan")

    # --- Complexity (direction: -1, entropy of normalized abs attribution) ---
    try:
        cx = quantus.Complexity(
            abs=True,
            normalise=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = cx(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        )
        results["complexity"] = float(scores[0])
    except Exception as exc:
        logger.warning("Complexity failed: %s: %s", type(exc).__name__, exc)
        results["complexity"] = float("nan")

    # =====================================================================
    # RANDOMIZATION
    # =====================================================================

    # --- ModelParameterRandomisation (direction: -1) ---
    # Returns correlation between original and randomised-model explanations.
    # Lower = better (more different from random → more dependent on params).
    try:
        mprt = quantus.ModelParameterRandomisation(
            layer_order="top_down",
            normalise=True,
            abs=True,
            return_average_correlation=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = mprt(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            explain_func=ef,
            explain_func_kwargs=ef_kwargs,
            device=device,
        )
        results["model_parameter_randomisation"] = float(scores[0])
    except Exception as exc:
        logger.warning("ModelParameterRandomisation failed: %s: %s", type(exc).__name__, exc)
        results["model_parameter_randomisation"] = float("nan")

    # --- RandomLogit (direction: -1) ---
    # Correlation between attribution for true target and random target.
    # Lower = better (explanation changes when target changes).
    try:
        rl = quantus.RandomLogit(
            num_classes=num_classes,
            abs=True,
            normalise=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = rl(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            explain_func=ef,
            explain_func_kwargs=ef_kwargs,
            device=device,
        )
        results["random_logit"] = float(scores[0])
    except Exception as exc:
        logger.warning("RandomLogit failed: %s: %s", type(exc).__name__, exc)
        results["random_logit"] = float("nan")

    # =====================================================================
    # AXIOMATIC
    # =====================================================================

    # --- Completeness (direction: -1) ---
    # |sum(attr) - (f(x) - f(baseline))|. Only meaningful for methods that
    # claim completeness: IG, DeepLift, LRP. For others, return NaN.
    if fae_method is not None and fae_method not in _COMPLETENESS_METHODS:
        logger.debug(
            "Completeness skipped for '%s' (not a completeness-satisfying method).",
            fae_method,
        )
        results["completeness"] = float("nan")
    else:
        try:
            comp = quantus.Completeness(
                abs=False,
                normalise=False,
                perturb_baseline="black",
                return_aggregate=False,
                disable_warnings=True,
            )
            scores = comp(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                channel_first=True,
                device=device,
            )
            results["completeness"] = float(scores[0])
        except Exception as exc:
            logger.warning("Completeness failed: %s: %s", type(exc).__name__, exc)
            results["completeness"] = float("nan")

    # --- NonSensitivity (direction: -1) ---
    # NonSensitivity checks if attributions change when features are
    # perturbed. With features_in_step=1 (only correct value due to a
    # Quantus shape bug), it runs one forward pass per feature.
    # At 224x224x3 that is 150k passes — infeasible per image.
    # Workaround: downsample both input and attribution to 56x56 for
    # this metric only (9408 features → ~9k passes, ~34s per image).
    # Skipped by default; enable via include_non_sensitivity=True.
    if not include_non_sensitivity:
        results["non_sensitivity"] = float("nan")
        return results

    try:
        from scipy.ndimage import zoom

        scale = 56 / image.shape[-1]  # 56/224 = 0.25
        x_small = zoom(image, (1, scale, scale), order=1)           # (3, 56, 56)
        a_small = zoom(attribution, (1, scale, scale), order=1)     # (3, 56, 56)
        x_small_batch = x_small[np.newaxis, ...]                    # (1, 3, 56, 56)
        a_small_batch = a_small[np.newaxis, ...]                    # (1, 3, 56, 56)

        ns = quantus.NonSensitivity(
            features_in_step=1,
            abs=True,
            normalise=True,
            perturb_baseline="black",
            return_aggregate=False,
            disable_warnings=True,
        )
        scores = ns(
            model=model,
            x_batch=x_small_batch,
            y_batch=y_batch,
            a_batch=a_small_batch,
            channel_first=True,
            explain_func=ef,
            explain_func_kwargs=ef_kwargs,
            device=device,
        )
        results["non_sensitivity"] = float(scores[0])
    except Exception as exc:
        logger.warning("NonSensitivity failed: %s: %s", type(exc).__name__, exc)
        results["non_sensitivity"] = float("nan")

    return results
