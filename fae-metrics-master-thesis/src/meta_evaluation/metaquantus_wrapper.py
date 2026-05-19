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
import datetime
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Pre-screening
# ---------------------------------------------------------------------------

def screen_meta_eval_candidates(
    vertical_slice_df: pd.DataFrame,
    min_valid_fraction: float = 0.5,
) -> pd.DataFrame:
    """Determine which (model, fae_method, metric) triples should run meta-evaluation.

    Skips triples where the underlying metric is mostly NaN in the input data,
    which would make NR and AR scores meaningless and waste Colab GPU time.

    Parameters
    ----------
    vertical_slice_df : pd.DataFrame
        Long-format DataFrame with columns: model, image_id, fae_method,
        metric, score.  Typically ``results/vertical_slice_7fae_12metrics.csv``.
    min_valid_fraction : float
        Minimum fraction of non-NaN scores required for a triple to be
        eligible.  Default ``0.5`` — triples with fewer than half their
        images producing a valid score are skipped.

    Returns
    -------
    pd.DataFrame
        Columns: model, fae_method, metric, n_valid_images, total_images,
        valid_fraction, run_meta_eval (bool).
        One row per unique (model, fae_method, metric) triple found in
        *vertical_slice_df*.  ``run_meta_eval`` is True iff
        ``valid_fraction >= min_valid_fraction``.

    Examples
    --------
    MPR × GradCAM × SqueezeNet has 0/12 valid scores in the 12-image slice
    (100% NaN due to SqueezeNet architecture incompatibility) →
    ``run_meta_eval = False``.
    """
    rows: list[dict] = []
    for (model, fae, metric), grp in vertical_slice_df.groupby(
        ["model", "fae_method", "metric"]
    ):
        total = len(grp)
        n_valid = int(grp["score"].notna().sum())
        valid_fraction = n_valid / total if total > 0 else 0.0
        rows.append(
            {
                "model": model,
                "fae_method": fae,
                "metric": metric,
                "n_valid_images": n_valid,
                "total_images": total,
                "valid_fraction": valid_fraction,
                "run_meta_eval": valid_fraction >= min_valid_fraction,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model", "fae_method", "metric", "n_valid_images",
                "total_images", "valid_fraction", "run_meta_eval",
            ]
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full orchestrator
# ---------------------------------------------------------------------------

def run_meta_evaluation_full(
    vertical_slice_df: pd.DataFrame,
    models: dict[str, nn.Module],
    metric_fns: dict[str, Callable],
    fae_methods: dict[str, Callable],
    images: list,
    targets: list[int],
    device: str = "cuda",
    n_seeds: int = 5,
    n_levels: int = 5,
    output_csv: str = "results/meta_evaluation_reliability.csv",
    progress_log: str | None = "results/meta_eval_progress.log",
) -> pd.DataFrame:
    """Orchestrate the full meta-evaluation over all eligible (model, fae, metric) triples.

    Runs :func:`screen_meta_eval_candidates`, then iterates over every eligible
    triple calling :func:`meta_evaluate_metric`.  Rows are appended to
    *output_csv* after each (model, fae) pair completes so that a Colab session
    timeout does not lose all progress.

    Resume behaviour: if *output_csv* already exists, triples that have a
    ``status`` of ``'completed'`` or ``'skipped_nan'`` are skipped.

    Parameters
    ----------
    vertical_slice_df : pd.DataFrame
        Long-format results used by :func:`screen_meta_eval_candidates`.
    models : dict[str, nn.Module]
        Mapping from model name to a loaded, eval-mode classifier.
    metric_fns : dict[str, Callable]
        Mapping from metric name to a pre-configured Quantus metric instance.
    fae_methods : dict[str, Callable]
        Mapping from FAE method name to an attribution function with Quantus
        signature ``(model, inputs, targets, **kwargs) -> np.ndarray``.
    images : list of array-like or torch.Tensor
        Test images, each of shape ``(C, H, W)``.
    targets : list of int
        Target class indices paired with *images*.
    device : str
        Compute device.  Default ``'cuda'`` for Colab; use ``'cpu'`` locally.
    n_seeds : int
        NR perturbation seeds.
    n_levels : int
        AR degradation levels.
    output_csv : str
        Path for incremental CSV output.  Parent directory is created if needed.
    progress_log : str or None
        Path for a timestamped progress log.  Set to ``None`` to disable.
        Each completed or failed triple appends one line.

    Returns
    -------
    pd.DataFrame
        Columns: model, fae_method, metric, nr_score, ar_score,
        combined_reliability, n_valid_inputs, runtime_seconds, status.
        ``status`` is one of ``'completed'``, ``'skipped_nan'``, ``'failed'``.
    """
    _RESULT_COLS = [
        "model", "fae_method", "metric", "nr_score", "ar_score",
        "combined_reliability", "n_valid_inputs", "runtime_seconds", "status",
    ]

    candidates = screen_meta_eval_candidates(vertical_slice_df)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(progress_log) if progress_log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect triples already written with a terminal status
    done_triples: set[tuple[str, str, str]] = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            for _, row in existing.iterrows():
                if row.get("status") in ("completed", "skipped_nan"):
                    done_triples.add(
                        (str(row["model"]), str(row["fae_method"]), str(row["metric"]))
                    )
            if done_triples:
                logger.info("Resume: %d triples already done.", len(done_triples))
        except Exception as exc:
            logger.warning("Could not read existing CSV for resume: %s", exc)

    imgs_np = [_to_numpy(img) for img in images]
    x_stack = np.stack(imgs_np)            # (B, C, H, W)
    y_array = np.array(targets, dtype=np.int64)

    all_rows: list[dict] = []

    for (model_name, fae_name), group_df in candidates.groupby(
        ["model", "fae_method"]
    ):
        if model_name not in models:
            logger.warning("Model '%s' not found in models dict; skipping.", model_name)
            continue
        if fae_name not in fae_methods:
            logger.warning("FAE '%s' not found in fae_methods dict; skipping.", fae_name)
            continue

        model = models[model_name]
        explain_fn = fae_methods[fae_name]

        # Compute baseline attributions for this (model, fae) pair once
        try:
            attrs_batch = explain_fn(model, x_stack, y_array)  # (B, C, H, W)
            attributions = [attrs_batch[i] for i in range(len(imgs_np))]
        except Exception as exc:
            logger.error(
                "Attribution failed for (%s, %s): %s — using zeros.",
                model_name, fae_name, exc,
            )
            attributions = [np.zeros_like(img) for img in imgs_np]

        group_rows: list[dict] = []

        for _, cand_row in group_df.iterrows():
            metric_name = str(cand_row["metric"])
            run = bool(cand_row["run_meta_eval"])
            n_valid = int(cand_row["n_valid_images"])
            triple = (model_name, fae_name, metric_name)

            if triple in done_triples:
                logger.debug("Skipping already-done triple %s.", triple)
                continue

            t0 = time.perf_counter()

            if not run:
                status = "skipped_nan"
                nr_s = ar_s = combined = float("nan")
                elapsed = 0.0

            elif metric_name not in metric_fns:
                logger.warning("Metric '%s' not in metric_fns dict.", metric_name)
                status = "failed"
                nr_s = ar_s = combined = float("nan")
                elapsed = 0.0

            else:
                try:
                    result = meta_evaluate_metric(
                        metric_name=metric_name,
                        metric_fn=metric_fns[metric_name],
                        model=model,
                        images=imgs_np,
                        attributions=attributions,
                        targets=targets,
                        explain_fn=explain_fn,
                        device=device,
                        n_seeds=n_seeds,
                        n_levels=n_levels,
                    )
                    nr_s = result["nr"]["nr_score"]
                    ar_s = result["ar"]["ar_score"]
                    combined = result["combined_reliability"]
                    status = "completed"
                except Exception as exc:
                    logger.error(
                        "meta_evaluate_metric failed (%s, %s, %s): %s",
                        model_name, fae_name, metric_name, exc,
                    )
                    nr_s = ar_s = combined = float("nan")
                    status = "failed"

                elapsed = time.perf_counter() - t0

            out_row = {
                "model": model_name,
                "fae_method": fae_name,
                "metric": metric_name,
                "nr_score": nr_s,
                "ar_score": ar_s,
                "combined_reliability": combined,
                "n_valid_inputs": n_valid,
                "runtime_seconds": round(elapsed, 3),
                "status": status,
            }
            group_rows.append(out_row)

            if log_path and status != "skipped_nan":
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                with open(log_path, "a") as lf:
                    lf.write(
                        f"{ts} | {model_name:15s} | {fae_name:25s} | "
                        f"{metric_name:40s} | {status:12s} | {elapsed:.1f}s\n"
                    )

        # Incremental write: append this (model, fae) pair's rows to CSV
        if group_rows:
            batch_df = pd.DataFrame(group_rows)
            write_header = not output_path.exists()
            batch_df.to_csv(output_path, mode="a", header=write_header, index=False)
            logger.info(
                "Appended %d rows for (%s, %s) → %s",
                len(group_rows), model_name, fae_name, output_csv,
            )
            all_rows.extend(group_rows)

    if all_rows:
        return pd.DataFrame(all_rows)
    return pd.DataFrame(columns=_RESULT_COLS)
