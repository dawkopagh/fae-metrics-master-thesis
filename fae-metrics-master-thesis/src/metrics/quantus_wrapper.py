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
import time
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
    include_model_parameter_randomisation: bool = True,
    return_timings: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    """Compute evaluation metrics for a single image attribution.

    By default computes 11 metrics. NonSensitivity is excluded because
    it requires ~9k forward passes per image (Quantus features_in_step
    bug forces features_in_step=1). Set ``include_non_sensitivity=True``
    to include it at the cost of significant runtime.

    ``ModelParameterRandomisation`` is computed by default but can be
    disabled with ``include_model_parameter_randomisation=False`` (records
    NaN, preserving the metric schema). On ISIC 2017 with Quantus 0.6.0 it
    raises an AssertionError on every sample and emits a per-call deprecation
    notice; disabling it removes that noise and wasted compute. The
    Randomization category is still represented by RandomLogit.

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
    dict[str, float] or tuple[dict[str, float], dict[str, float]]
        Keys are metric snake_case names. Values are floats or
        ``float('nan')`` on failure or intentional skip (Completeness
        on non-completeness methods). If ``return_timings=True``, also
        returns a dict mapping metric name to elapsed seconds.
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
    timings: dict[str, float] = {}

    # =====================================================================
    # FAITHFULNESS
    # =====================================================================

    # --- FaithfulnessCorrelation (direction: +1) ---
    t0 = time.perf_counter()
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
    finally:
        timings["faithfulness_correlation"] = time.perf_counter() - t0

    # --- PixelFlipping (direction: +1, AUC — higher means more faithful) ---
    t0 = time.perf_counter()
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
    finally:
        timings["pixel_flipping"] = time.perf_counter() - t0

    # =====================================================================
    # ROBUSTNESS
    # =====================================================================

    # --- MaxSensitivity (direction: -1) ---
    t0 = time.perf_counter()
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
    finally:
        timings["max_sensitivity"] = time.perf_counter() - t0

    # --- AvgSensitivity (direction: -1) ---
    t0 = time.perf_counter()
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
    finally:
        timings["avg_sensitivity"] = time.perf_counter() - t0

    # =====================================================================
    # LOCALIZATION
    # =====================================================================

    # --- RelevanceMassAccuracy (direction: +1) ---
    t0 = time.perf_counter()
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
    finally:
        timings["relevance_mass_accuracy"] = time.perf_counter() - t0

    # --- PointingGame (direction: +1) ---
    t0 = time.perf_counter()
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
    finally:
        timings["pointing_game"] = time.perf_counter() - t0

    # =====================================================================
    # COMPLEXITY
    # =====================================================================

    # --- Sparseness (direction: +1, Gini-like) ---
    t0 = time.perf_counter()
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
    finally:
        timings["sparseness"] = time.perf_counter() - t0

    # --- Complexity (direction: -1, entropy of normalized abs attribution) ---
    t0 = time.perf_counter()
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
    finally:
        timings["complexity"] = time.perf_counter() - t0

    # =====================================================================
    # RANDOMIZATION
    # =====================================================================

    # --- ModelParameterRandomisation (direction: -1) ---
    # Returns correlation between original and randomised-model explanations.
    # Lower = better (more different from random → more dependent on params).
    #
    # Excluded via include_model_parameter_randomisation=False on ISIC 2017:
    # Quantus 0.6.0's MPRT raises AssertionError on every (model, image) here
    # (it also emits a deprecation notice per call), so it contributes only
    # NaN and floods the log. The Randomization category is still covered by
    # RandomLogit below. When disabled we record NaN to keep the 12-metric
    # schema stable (mirrors the NonSensitivity handling).
    if not include_model_parameter_randomisation:
        t0 = time.perf_counter()
        results["model_parameter_randomisation"] = float("nan")
        timings["model_parameter_randomisation"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
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
        finally:
            timings["model_parameter_randomisation"] = time.perf_counter() - t0

    # --- RandomLogit (direction: -1) ---
    # Correlation between attribution for true target and random target.
    # Lower = better (explanation changes when target changes).
    t0 = time.perf_counter()
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
    finally:
        timings["random_logit"] = time.perf_counter() - t0

    # =====================================================================
    # AXIOMATIC
    # =====================================================================

    # --- Completeness (direction: -1) ---
    # |sum(attr) - (f(x) - f(baseline))|. Only meaningful for methods that
    # claim completeness: IG, DeepLift, LRP. For others, return NaN.
    if fae_method is not None and fae_method not in _COMPLETENESS_METHODS:
        t0 = time.perf_counter()
        logger.debug(
            "Completeness skipped for '%s' (not a completeness-satisfying method).",
            fae_method,
        )
        results["completeness"] = float("nan")
        timings["completeness"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
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
        finally:
            timings["completeness"] = time.perf_counter() - t0

    # --- NonSensitivity (direction: -1) ---
    # NonSensitivity checks if attributions change when features are
    # perturbed. With features_in_step=1 (only correct value due to a
    # Quantus shape bug), it runs one forward pass per feature.
    # At 224x224x3 that is 150k passes — infeasible per image.
    # Workaround: downsample both input and attribution to 56x56 for
    # this metric only (9408 features → ~9k passes, ~34s per image).
    # Skipped by default; enable via include_non_sensitivity=True.
    if not include_non_sensitivity:
        t0 = time.perf_counter()
        results["non_sensitivity"] = float("nan")
        timings["non_sensitivity"] = time.perf_counter() - t0
        if return_timings:
            return results, timings
        return results

    t0 = time.perf_counter()
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
    finally:
        timings["non_sensitivity"] = time.perf_counter() - t0

    if return_timings:
        return results, timings

    return results


def compute_all_metrics_batched(
    model: nn.Module,
    images: np.ndarray,
    attributions: np.ndarray,
    targets,
    masks: Optional[np.ndarray] = None,
    device: str = "cpu",
    explain_func: Optional[Callable] = None,
    explain_func_kwargs: Optional[dict] = None,
    fae_method: Optional[str] = None,
    num_classes: int = 3,
    include_non_sensitivity: bool = False,
    include_model_parameter_randomisation: bool = True,
    return_timings: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], dict[str, float]]:
    """Compute evaluation metrics for a BATCH of image attributions at once.

    This is the multi-image counterpart to :func:`compute_all_metrics`. It
    calls each Quantus metric ONCE with a full ``(B, ...)`` batch (rather than
    B separate ``(1, ...)`` calls) so the GPU is saturated, then distributes
    the B per-sample scores Quantus returns into B independent result dicts.

    The metric constructor configs are IDENTICAL to the single-image function
    (nr_runs, subset_size, nr_samples, features_in_step, perturb_baseline,
    normalise, abs, ...). For deterministic metrics this yields per-sample
    scores numerically equal to the single-image path; for stochastic metrics
    (FaithfulnessCorrelation, Max/AvgSensitivity, RandomLogit) the batched run
    draws a different but equally valid random sample. NOTE: not yet covered
    by unit tests and not used by the reported full run (which used the
    single-image path); validate against ``compute_all_metrics`` before
    relying on it for a future re-run.

    ``fae_method`` is assumed CONSTANT across the batch (the caller batches per
    ``(model, fae)`` pair), so Completeness applicability, the NonSensitivity
    skip, and the MPR include-flag are decided ONCE for the whole batch.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    images : np.ndarray
        Input batch of shape ``(B, 3, H, W)`` (channel-first).
    attributions : np.ndarray
        Attribution batch of shape ``(B, 3, H, W)`` (channel-first).
    targets : sequence of int
        Length-B target class indices.
    masks : np.ndarray or None
        Binary segmentation masks of shape ``(B, 1, H, W)`` or ``None``.
        Used by RelevanceMassAccuracy and PointingGame.
    device : str
        Device for computation.
    explain_func : callable or None
        Batched explain_func (Quantus passes perturbed batches). Required by
        Max/AvgSensitivity, ModelParameterRandomisation, RandomLogit.
    explain_func_kwargs : dict or None
        Extra kwargs forwarded to *explain_func*.
    fae_method : str or None
        FAE method that produced *attributions* (constant across the batch).
    num_classes : int
        Number of output classes (for RandomLogit). Default 3.
    include_non_sensitivity : bool
        Include the (slow) NonSensitivity metric. Default False.
    include_model_parameter_randomisation : bool
        Include ModelParameterRandomisation. Default True.
    return_timings : bool
        If True, also return per-metric wall times for the whole batch.

    Returns
    -------
    list[dict[str, float]] or tuple[list[dict[str, float]], dict[str, float]]
        A list of B dicts, each identical in shape to the dict returned by
        :func:`compute_all_metrics` (metric snake_case name -> float). If
        ``return_timings=True``, also returns a dict mapping metric name to
        elapsed seconds for the whole-batch call.
    """
    images = np.asarray(images)
    attributions = np.asarray(attributions)
    B = images.shape[0]

    x_batch = images                              # (B, 3, H, W)
    a_batch = attributions                        # (B, 3, H, W)
    y_batch = np.asarray([int(t) for t in targets])  # (B,)

    s_batch: Optional[np.ndarray] = None
    if masks is not None:
        s_batch = np.asarray(masks)               # (B, 1, H, W)

    # Single-channel attribution for localization metrics
    a_batch_1ch = a_batch.sum(axis=1, keepdims=True)  # (B, 1, H, W)

    ef = explain_func
    ef_kwargs = explain_func_kwargs or {}

    # One result dict per sample; one timings dict shared for the batch.
    results: list[dict[str, float]] = [{} for _ in range(B)]
    timings: dict[str, float] = {}

    def _set_all(metric_name: str, value: float) -> None:
        """Assign the same scalar (e.g. NaN) to every sample for a metric."""
        for i in range(B):
            results[i][metric_name] = float(value)

    def _distribute(metric_name: str, scores) -> None:
        """Distribute Quantus's length-B score vector across the B dicts."""
        arr = np.asarray(scores, dtype=float).reshape(-1)
        if arr.shape[0] != B:
            # Defensive: Quantus must return exactly B scores. If not, fail the
            # whole metric (NaN for all) rather than silently mis-aligning.
            raise ValueError(
                f"{metric_name}: expected {B} scores, got {arr.shape[0]}"
            )
        for i in range(B):
            results[i][metric_name] = float(arr[i])

    def _run(metric_name: str, call: Callable[[], object]) -> None:
        """Run one batched metric call; NaN-fill all samples on failure."""
        t0 = time.perf_counter()
        try:
            _distribute(metric_name, call())
        except Exception as exc:  # noqa: BLE001 — mirror single-image behavior
            logger.warning(
                "%s (batched) failed: %s: %s", metric_name, type(exc).__name__, exc
            )
            _set_all(metric_name, float("nan"))
        finally:
            timings[metric_name] = time.perf_counter() - t0

    # =====================================================================
    # FAITHFULNESS
    # =====================================================================

    # --- FaithfulnessCorrelation (direction: +1) [STOCHASTIC] ---
    def _fc():
        fc = quantus.FaithfulnessCorrelation(
            nr_runs=100,
            subset_size=224,
            perturb_baseline="black",
            normalise=True,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        return fc(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, device=device,
        )
    _run("faithfulness_correlation", _fc)

    # --- PixelFlipping (direction: +1, AUC) [DETERMINISTIC] ---
    def _pf():
        pf = quantus.PixelFlipping(
            features_in_step=224,
            perturb_baseline="black",
            normalise=True,
            abs=False,
            return_aggregate=False,
            return_auc_per_sample=True,
            disable_warnings=True,
        )
        return pf(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, device=device,
        )
    _run("pixel_flipping", _pf)

    # =====================================================================
    # ROBUSTNESS
    # =====================================================================

    # --- MaxSensitivity (direction: -1) [STOCHASTIC] ---
    def _ms():
        ms = quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            normalise=False,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        return ms(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, explain_func=ef, explain_func_kwargs=ef_kwargs,
            device=device,
        )
    _run("max_sensitivity", _ms)

    # --- AvgSensitivity (direction: -1) [STOCHASTIC] ---
    def _avgs():
        avgs = quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            normalise=False,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        return avgs(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, explain_func=ef, explain_func_kwargs=ef_kwargs,
            device=device,
        )
    _run("avg_sensitivity", _avgs)

    # =====================================================================
    # LOCALIZATION
    # =====================================================================

    # --- RelevanceMassAccuracy (direction: +1) [DETERMINISTIC] ---
    def _rma():
        rma = quantus.RelevanceMassAccuracy(
            normalise=True,
            abs=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        return rma(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_1ch,
            s_batch=s_batch, channel_first=True, device=device,
        )
    _run("relevance_mass_accuracy", _rma)

    # --- PointingGame (direction: +1) [DETERMINISTIC] ---
    def _pg():
        pg = quantus.PointingGame(
            normalise=True,
            abs=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        return pg(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_1ch,
            s_batch=s_batch, channel_first=True, device=device,
        )
    _run("pointing_game", _pg)

    # =====================================================================
    # COMPLEXITY
    # =====================================================================

    # --- Sparseness (direction: +1, Gini-like) [DETERMINISTIC] ---
    def _sp():
        sp = quantus.Sparseness(
            abs=True,
            normalise=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        return sp(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, device=device,
        )
    _run("sparseness", _sp)

    # --- Complexity (direction: -1, entropy) [DETERMINISTIC] ---
    def _cx():
        cx = quantus.Complexity(
            abs=True,
            normalise=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        return cx(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, device=device,
        )
    _run("complexity", _cx)

    # =====================================================================
    # RANDOMIZATION
    # =====================================================================

    # --- ModelParameterRandomisation (direction: -1) ---
    # Skipped (NaN for all samples) unless explicitly enabled — mirrors the
    # single-image path and keeps the 12-metric schema stable.
    if not include_model_parameter_randomisation:
        t0 = time.perf_counter()
        _set_all("model_parameter_randomisation", float("nan"))
        timings["model_parameter_randomisation"] = time.perf_counter() - t0
    else:
        def _mprt():
            mprt = quantus.ModelParameterRandomisation(
                layer_order="top_down",
                normalise=True,
                abs=True,
                return_average_correlation=True,
                return_aggregate=False,
                disable_warnings=True,
            )
            return mprt(
                model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
                channel_first=True, explain_func=ef, explain_func_kwargs=ef_kwargs,
                device=device,
            )
        _run("model_parameter_randomisation", _mprt)

    # --- RandomLogit (direction: -1) [STOCHASTIC — random target class] ---
    def _rl():
        rl = quantus.RandomLogit(
            num_classes=num_classes,
            abs=True,
            normalise=True,
            return_aggregate=False,
            disable_warnings=True,
        )
        return rl(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
            channel_first=True, explain_func=ef, explain_func_kwargs=ef_kwargs,
            device=device,
        )
    _run("random_logit", _rl)

    # =====================================================================
    # AXIOMATIC
    # =====================================================================

    # --- Completeness (direction: -1) [DETERMINISTIC] ---
    # fae_method is constant across the batch, so applicability is decided once.
    if fae_method is not None and fae_method not in _COMPLETENESS_METHODS:
        t0 = time.perf_counter()
        logger.debug(
            "Completeness skipped for '%s' (not a completeness-satisfying method).",
            fae_method,
        )
        _set_all("completeness", float("nan"))
        timings["completeness"] = time.perf_counter() - t0
    else:
        def _comp():
            comp = quantus.Completeness(
                abs=False,
                normalise=False,
                perturb_baseline="black",
                return_aggregate=False,
                disable_warnings=True,
            )
            return comp(
                model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
                channel_first=True, device=device,
            )
        _run("completeness", _comp)

    # --- NonSensitivity (direction: -1) ---
    # Skipped (NaN for all samples) by default. When enabled, downsample each
    # sample to 56x56 (same workaround as the single-image path) and run once
    # over the batch.
    if not include_non_sensitivity:
        t0 = time.perf_counter()
        _set_all("non_sensitivity", float("nan"))
        timings["non_sensitivity"] = time.perf_counter() - t0
        if return_timings:
            return results, timings
        return results

    def _ns():
        from scipy.ndimage import zoom

        scale = 56 / images.shape[-1]
        x_small = np.stack(
            [zoom(images[i], (1, scale, scale), order=1) for i in range(B)], axis=0
        )  # (B, 3, 56, 56)
        a_small = np.stack(
            [zoom(attributions[i], (1, scale, scale), order=1) for i in range(B)], axis=0
        )  # (B, 3, 56, 56)
        ns = quantus.NonSensitivity(
            features_in_step=1,
            abs=True,
            normalise=True,
            perturb_baseline="black",
            return_aggregate=False,
            disable_warnings=True,
        )
        return ns(
            model=model, x_batch=x_small, y_batch=y_batch, a_batch=a_small,
            channel_first=True, explain_func=ef, explain_func_kwargs=ef_kwargs,
            device=device,
        )
    _run("non_sensitivity", _ns)

    if return_timings:
        return results, timings

    return results
