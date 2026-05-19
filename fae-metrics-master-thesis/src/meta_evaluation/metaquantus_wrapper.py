"""
Re-implementation of MetaQuantus NR and AR tests using Quantus primitives.

Follows Hedström, Höhne, Lapuschkin (2024) "MetaQuantus: A Meta-Evaluation
Framework for Assessing the Reliability of Explainable AI Metrics". TMLR.
No dependency on the metaquantus package.

Responsibilities (see docs/thesis_plan.md §2, Contribution 2; §6, Meta-Evaluation):
    - Noise Resilience (NR): measure metric stability under small input/weight noise
    - Adversarial Reactivity (AR): measure metric sensitivity to attribution degradation
    - Combined meta_evaluate_metric convenience entry-point

Not responsible for: metric selection (see metric_selection.py),
base metric computation (see metrics/quantus_wrapper.py),
or final aggregation (see aggregation/).
"""

from __future__ import annotations

import copy
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: object) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def _call_metric(
    metric_fn: Callable,
    model: nn.Module,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: np.ndarray,
    device: str,
) -> float:
    """Call a pre-configured Quantus metric instance; return mean batch score."""
    try:
        scores = metric_fn(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            channel_first=True,
            device=device,
        )
        valid = [float(s) for s in scores if not np.isnan(float(s))]
        return float(np.mean(valid)) if valid else float("nan")
    except Exception as exc:
        logger.warning("metric_fn call failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Noise Resilience (NR)
# ---------------------------------------------------------------------------

def noise_resilience_test(
    metric_fn: Callable,
    model: nn.Module,
    images: list,
    attributions: list,
    targets: list[int],
    explain_fn: Callable,
    perturbation_type: str = "input",
    n_seeds: int = 5,
    noise_std: float = 0.01,
    device: str = "cpu",
) -> dict:
    """Noise Resilience (NR) meta-evaluation test.

    Measures metric score stability under small, semantically meaningless
    perturbations (Hedström et al., 2024, §3.1).  A reliable metric has low
    score variance across noise seeds because the perturbation carries no
    explanatory information.

    Perturbation types:

    - ``'input'``: Gaussian noise N(0, ``noise_std``) added to each image.
      Attributions are recomputed via *explain_fn* on the noisy input.
      Metric is evaluated on (noisy input, recomputed attribution).

    - ``'weights'``: Gaussian noise N(0, ``noise_std``) added to all model
      parameters.  Attributions are recomputed via *explain_fn* with the
      noisy model.  Metric is evaluated on (original input, recomputed
      attribution) using the noisy model.

    Parameters
    ----------
    metric_fn : Callable
        Pre-configured Quantus metric instance.  Must accept keyword arguments
        ``model, x_batch, y_batch, a_batch, channel_first, device``.
    model : nn.Module
        Classifier in eval mode.
    images : list of array-like
        Test images, each of shape ``(C, H, W)``.
    attributions : list of array-like
        Pre-computed attribution maps paired with *images*, each ``(C, H, W)``.
        Unused during the test (recomputed under noise) but kept for interface
        consistency with :func:`adversarial_reactivity_test`.
    targets : list of int
        Target class indices paired with *images*.
    explain_fn : Callable
        Attribution function with Quantus signature:
        ``(model, inputs, targets, **kwargs) -> np.ndarray``
        where *inputs* is ``(B, C, H, W)`` float32, *targets* is ``(B,)`` int,
        and the return is ``(B, C, H, W)`` attributions.
    perturbation_type : {'input', 'weights'}
        Where to apply noise.  Default ``'input'``.
    n_seeds : int
        Number of independent noise realizations.
    noise_std : float
        Standard deviation of the Gaussian noise.
    device : str
        Device string passed to *metric_fn*.

    Returns
    -------
    dict
        ``mean_score`` : float — mean metric score across seeds.
        ``std_score``  : float — standard deviation across seeds.
        ``cv_score``   : float — coefficient of variation (std / |mean|).
        ``nr_score``   : float — reliability in (0, 1]:
            ``nr_score = 1 / (1 + cv_score)``.
            1.0 = perfectly stable; near 0 = highly unstable.
        ``raw_scores`` : list[float] — per-seed metric scores.
    """
    if perturbation_type not in ("input", "weights"):
        raise ValueError(
            f"perturbation_type must be 'input' or 'weights', "
            f"got '{perturbation_type}'"
        )

    imgs_np = [_to_numpy(img) for img in images]
    y_batch_base = np.array(targets, dtype=np.int64)

    raw_scores: list[float] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        if perturbation_type == "input":
            x_perturbed = np.stack([
                img + rng.standard_normal(img.shape).astype(np.float32) * noise_std
                for img in imgs_np
            ])  # (B, C, H, W)
            a_recomputed = explain_fn(model, x_perturbed, y_batch_base)
            eval_model = model
            x_eval = x_perturbed

        else:  # "weights"
            noisy_model = copy.deepcopy(model)
            with torch.no_grad():
                for param in noisy_model.parameters():
                    param.add_(torch.randn_like(param) * noise_std)
            noisy_model.eval()

            x_stack = np.stack(imgs_np)  # (B, C, H, W)
            a_recomputed = explain_fn(noisy_model, x_stack, y_batch_base)
            eval_model = noisy_model
            x_eval = x_stack

        score = _call_metric(metric_fn, eval_model, x_eval, y_batch_base, a_recomputed, device)
        raw_scores.append(score)

    valid = [s for s in raw_scores if not np.isnan(s)]
    mean_score = float(np.mean(valid)) if valid else float("nan")
    std_score = float(np.std(valid, ddof=0)) if len(valid) > 1 else 0.0

    if np.isnan(mean_score) or abs(mean_score) < 1e-10:
        cv_score = float("inf") if std_score > 1e-10 else 0.0
    else:
        cv_score = std_score / abs(mean_score)

    nr_score = 1.0 / (1.0 + cv_score) if np.isfinite(cv_score) else 0.0

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "cv_score": cv_score,
        "nr_score": nr_score,
        "raw_scores": raw_scores,
    }


# ---------------------------------------------------------------------------
# Adversarial Reactivity (AR)
# ---------------------------------------------------------------------------

def adversarial_reactivity_test(
    metric_fn: Callable,
    model: nn.Module,
    images: list,
    attributions: list,
    targets: list[int],
    n_levels: int = 5,
    level_min: float = 0.0,
    level_max: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Adversarial Reactivity (AR) meta-evaluation test.

    Measures whether the metric declines monotonically as attributions are
    progressively degraded (Hedström et al., 2024, §3.2).  A reliable metric
    reacts to the loss of explanatory signal: as more attribution values are
    zeroed out, the score should change in a systematic direction.

    Expected sign of Spearman monotonicity:

    - ``higher-is-better`` metrics (``METRIC_DIRECTIONS == +1`` in
      ``src/aggregation/normalize.py``): degradation should *decrease* scores
      → expected Spearman ρ < 0.
    - ``lower-is-better`` metrics (``METRIC_DIRECTIONS == -1``): degradation
      should *increase* scores → expected Spearman ρ > 0.

    The ``ar_score`` uses ``|Spearman ρ|`` and is therefore direction-agnostic;
    it works for both metric conventions without hardcoding direction.

    Parameters
    ----------
    metric_fn : Callable
        Pre-configured Quantus metric instance.
    model : nn.Module
        Classifier in eval mode.
    images : list of array-like
        Test images, each ``(C, H, W)``.
    attributions : list of array-like
        Attribution maps, each ``(C, H, W)``.
    targets : list of int
        Target class indices.
    n_levels : int
        Number of degradation levels (includes endpoints).
    level_min : float
        Minimum degradation fraction (0.0 = original attribution).
    level_max : float
        Maximum degradation fraction (1.0 = fully randomised).
    device : str
        Device string passed to *metric_fn*.

    Returns
    -------
    dict
        ``levels``        : list[float] — degradation fractions evaluated.
        ``scores``        : list[float] — mean metric score at each level.
        ``monotonicity``  : float — Spearman ρ between levels and scores.
            0.0 when scores are constant (no structure to measure).
        ``ar_score``      : float — |monotonicity|, bounded in [0, 1].
            1.0 = perfectly monotonic; 0.0 = no systematic response.
    """
    imgs_np = [_to_numpy(img) for img in images]
    attrs_np = [_to_numpy(a) for a in attributions]

    x_batch = np.stack(imgs_np)         # (B, C, H, W)
    a_base = np.stack(attrs_np)         # (B, C, H, W)
    y_batch = np.array(targets, dtype=np.int64)

    levels = list(np.linspace(level_min, level_max, n_levels))
    scores: list[float] = []

    rng = np.random.default_rng(0)

    for frac in levels:
        if frac == 0.0:
            a_degraded = a_base.copy()
        else:
            a_degraded = a_base.copy()
            B, C, H, W = a_degraded.shape
            n_total = C * H * W
            n_zero = max(1, int(round(frac * n_total)))

            for b in range(B):
                flat = a_degraded[b].reshape(-1)
                idx = rng.choice(n_total, size=n_zero, replace=False)
                flat[idx] = 0.0
                a_degraded[b] = flat.reshape(C, H, W)

        score = _call_metric(metric_fn, model, x_batch, y_batch, a_degraded, device)
        scores.append(score)

    valid_mask = [not np.isnan(s) for s in scores]
    valid_levels = [lv for lv, ok in zip(levels, valid_mask) if ok]
    valid_scores = [s for s, ok in zip(scores, valid_mask) if ok]

    if len(valid_scores) >= 2 and np.std(valid_scores) > 1e-12:
        monotonicity = float(spearmanr(valid_levels, valid_scores).statistic)
    elif len(valid_scores) >= 2:
        # Constant scores → no monotonic structure
        monotonicity = 0.0
    else:
        monotonicity = float("nan")

    ar_score = abs(monotonicity) if not np.isnan(monotonicity) else float("nan")

    return {
        "levels": levels,
        "scores": scores,
        "monotonicity": monotonicity,
        "ar_score": ar_score,
    }


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def meta_evaluate_metric(
    metric_name: str,
    metric_fn: Callable,
    model: nn.Module,
    images: list,
    attributions: list,
    targets: list[int],
    explain_fn: Callable,
    device: str = "cpu",
    n_seeds: int = 5,
    n_levels: int = 5,
) -> dict:
    """Run NR and AR tests for a single metric; return a combined reliability report.

    Parameters
    ----------
    metric_name : str
        Human-readable metric identifier (used only as the ``'metric'`` key).
    metric_fn : Callable
        Pre-configured Quantus metric instance.
    model : nn.Module
        Classifier in eval mode.
    images : list of array-like
        Test images, each ``(C, H, W)``.
    attributions : list of array-like
        Pre-computed attribution maps paired with *images*, each ``(C, H, W)``.
    targets : list of int
        Target class indices.
    explain_fn : Callable
        Attribution function: ``(model, inputs, targets, **kwargs) -> np.ndarray``.
    device : str
        Compute device.
    n_seeds : int
        Number of NR perturbation seeds.
    n_levels : int
        Number of AR degradation levels.

    Returns
    -------
    dict
        ``metric``               : str   — metric name.
        ``nr``                   : dict  — :func:`noise_resilience_test` result.
        ``ar``                   : dict  — :func:`adversarial_reactivity_test` result.
        ``combined_reliability`` : float — ``0.5 * (nr_score + ar_score)``.
    """
    nr_result = noise_resilience_test(
        metric_fn=metric_fn,
        model=model,
        images=images,
        attributions=attributions,
        targets=targets,
        explain_fn=explain_fn,
        n_seeds=n_seeds,
        device=device,
    )
    ar_result = adversarial_reactivity_test(
        metric_fn=metric_fn,
        model=model,
        images=images,
        attributions=attributions,
        targets=targets,
        n_levels=n_levels,
        device=device,
    )

    nr_score = nr_result["nr_score"]
    ar_score = ar_result["ar_score"]

    if np.isnan(nr_score) or np.isnan(ar_score):
        combined = float("nan")
    else:
        combined = 0.5 * (nr_score + ar_score)

    return {
        "metric": metric_name,
        "nr": nr_result,
        "ar": ar_result,
        "combined_reliability": combined,
    }
