"""
End-to-end orchestrator for the multi-criteria FAE evaluation framework.

Responsibilities (see docs/thesis_plan.md §6, Framework Architecture):
    - Accept configuration and execute the evaluation pipeline
    - For the vertical slice: iterate over images × models × FAE methods,
      compute attributions, then evaluate with three Quantus metrics

Not responsible for: any individual pipeline stage — those are implemented
in their respective modules. This module contains only orchestration logic.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.attributions.generate import (
    FAE_METHODS,
    compute_gradcam,
    compute_integrated_gradients,
    compute_or_load,
    compute_saliency,
)
from src.attributions.cache import AttributionCache
from src.data.isic_dataset import ISIC2017Dataset
from src.metrics.quantus_wrapper import compute_three_metrics
from src.models.classifiers import (
    get_gradcam_target_layer,
    load_resnet18,
    load_squeezenet,
)

logger = logging.getLogger(__name__)


def _make_explain_func(
    fae_method: str,
    model: nn.Module,
    arch: str,
    device: str,
):
    """Build a Quantus-compatible explain_func for a given FAE method.

    Parameters
    ----------
    fae_method : str
        One of ``'integrated_gradients'``, ``'saliency'``, ``'gradcam'``.
    model : nn.Module
        The classifier (already on *device*, in eval mode).
    arch : str
        Architecture name (``'resnet18'`` or ``'squeezenet'``).
    device : str
        Device string.

    Returns
    -------
    callable
        Function with signature ``(model, inputs, targets, **kwargs) -> np.ndarray``.
    """

    def _explain(model, inputs, targets, **kwargs):
        """Quantus explain_func: compute attributions for a batch."""
        # inputs: np.ndarray (B, C, H, W), targets: np.ndarray (B,)
        attrs = []
        for i in range(inputs.shape[0]):
            img_t = torch.tensor(inputs[i], dtype=torch.float32)  # (C, H, W)
            tgt = int(targets[i])
            if fae_method == "integrated_gradients":
                a = compute_integrated_gradients(model, img_t, tgt, device=device)
            elif fae_method == "saliency":
                a = compute_saliency(model, img_t, tgt, device=device)
            elif fae_method == "gradcam":
                target_layer = get_gradcam_target_layer(model, arch)
                a = compute_gradcam(model, img_t, tgt, target_layer, device=device)
            else:
                raise ValueError(f"Unknown FAE method: {fae_method}")
            attrs.append(a.numpy())
        return np.stack(attrs, axis=0)

    return _explain


def run_vertical_slice(
    data_root: str = "data",
    resnet_weights: str = "resnet_skin.pth",
    squeezenet_weights: str = "squeezenet_skin.pth",
    split: str = "test",
    device: Optional[str] = None,
    output_csv: str = "results/vertical_slice.csv",
    seed: int = 42,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run the vertical-slice evaluation: 2 models × 3 FAE × 3 metrics.

    Iterates over every image in the requested split and, for each
    (image, model, FAE method) triple, computes the attribution and
    evaluates it with FaithfulnessCorrelation, MaxSensitivity, and
    RelevanceMassAccuracy.

    The **model's top-1 prediction** is used as the target class for
    attribution (not the ground-truth label). This is intentional:
    evaluation metrics assess explanation fidelity to the model's own
    decision function, which is the standard protocol in XAI evaluation
    literature (Hedström et al., 2023).

    Parameters
    ----------
    data_root : str
        Path to the data directory (containing ``images/`` and ``masks/``).
    resnet_weights : str
        Path to the ResNet-18 ``.pth`` weights file.
    squeezenet_weights : str
        Path to the SqueezeNet ``.pth`` weights file.
    split : str
        Dataset split to evaluate. Default ``'test'``.
    device : str or None
        Torch device. Auto-detected if None.
    output_csv : str
        Path for the output CSV. Parent directories are created if needed.
    seed : int
        Random seed for reproducibility.
    use_cache : bool
        If ``True`` (default), cache attribution maps to disk under
        ``attributions_cache/``. If ``False``, always recompute.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``model, image_id, fae_method, metric, score``.
    """
    # --- Reproducibility ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Seed: %d", device, seed)

    # --- Attribution cache ---
    cache: Optional[AttributionCache] = None
    if use_cache:
        cache = AttributionCache(root="attributions_cache")
        logger.info("Attribution cache enabled (root=%s)", cache.root)
    else:
        logger.info("Attribution cache disabled")

    # --- Load models ---
    models_dict: dict[str, tuple[nn.Module, str]] = {
        "resnet18": (load_resnet18(resnet_weights, device=device), "resnet18"),
        "squeezenet": (load_squeezenet(squeezenet_weights, device=device), "squeezenet"),
    }
    logger.info("Loaded %d models", len(models_dict))

    # --- Load dataset ---
    dataset = ISIC2017Dataset(
        root_dir=data_root, split=split, image_size=224, return_mask=True
    )
    logger.info("Dataset split='%s': %d images", split, len(dataset))

    fae_method_names = list(FAE_METHODS.keys())
    rows: list[dict] = []
    t_start = time.time()

    for img_idx in range(len(dataset)):
        sample = dataset[img_idx]
        image_tensor: torch.Tensor = sample["image"]      # (3, 224, 224)
        mask_tensor = sample["mask"]                        # (1, 224, 224) or None
        image_id: str = sample["image_id"]

        # Prepare numpy arrays for Quantus
        image_np = image_tensor.numpy()                     # (3, 224, 224)
        mask_np = mask_tensor.numpy() if mask_tensor is not None else None  # (1, 224, 224) or None

        for model_name, (model, arch) in models_dict.items():
            # Get model's top-1 prediction
            with torch.no_grad():
                logits = model(image_tensor.unsqueeze(0).to(device))
                target = int(logits.argmax(dim=1).item())

            for fae_name in fae_method_names:
                # Build compute kwargs and hyperparams for cache key
                if fae_name == "gradcam":
                    target_layer = get_gradcam_target_layer(model, arch)
                    fae_hyperparams: dict = {"image_size": 224}
                    compute_kwargs = dict(
                        model=model, image=image_tensor, target=target,
                        target_layer=target_layer, device=device,
                    )
                    compute_fn = compute_gradcam
                elif fae_name == "integrated_gradients":
                    fae_hyperparams = {"n_steps": 50}
                    compute_kwargs = dict(
                        model=model, image=image_tensor, target=target,
                        device=device, n_steps=50,
                    )
                    compute_fn = compute_integrated_gradients
                else:
                    fae_hyperparams = {}
                    compute_kwargs = dict(
                        model=model, image=image_tensor, target=target,
                        device=device,
                    )
                    compute_fn = FAE_METHODS[fae_name]

                attr_tensor = compute_or_load(
                    cache=cache,
                    compute_fn=compute_fn,
                    model_arch=arch,
                    fae_method=fae_name,
                    image_id=image_id,
                    target=target,
                    fae_hyperparams=fae_hyperparams,
                    **compute_kwargs,
                )

                attr_np = attr_tensor.numpy()  # (3, 224, 224)

                # Build explain_func for MaxSensitivity
                explain_func = _make_explain_func(fae_name, model, arch, device)

                # Compute metrics
                scores = compute_three_metrics(
                    model=model,
                    image=image_np,
                    attribution=attr_np,
                    mask=mask_np,
                    target=target,
                    device=device,
                    explain_func=explain_func,
                )

                for metric_name, score in scores.items():
                    rows.append(
                        {
                            "model": model_name,
                            "image_id": image_id,
                            "fae_method": fae_name,
                            "metric": metric_name,
                            "score": score,
                        }
                    )

                logger.info(
                    "[%d/%d] %s | %s | %s — done",
                    img_idx + 1,
                    len(dataset),
                    image_id,
                    model_name,
                    fae_name,
                )

    elapsed = time.time() - t_start
    logger.info("Vertical slice complete: %.1f s total", elapsed)

    df = pd.DataFrame(rows)

    # Write CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    return df
