"""
Attribution map generation for a configurable set of FAE methods via captum.

Responsibilities (see docs/thesis_plan.md §5, FAE Methods Evaluated):
    - Compute per-image attribution maps for Integrated Gradients, Saliency,
      and GradCAM (vertical-slice scope)
    - Return attributions in canonical shape (3, H, W) on CPU
    - Provide a FAE_METHODS dispatch dict for pipeline iteration

Not responsible for: caching attributions to disk (see cache.py),
ensembling multiple attribution maps (see ensembling.py),
or metric computation (see metrics/).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam, Saliency

from src.attributions.cache import AttributionCache


def compute_integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    n_steps: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute Integrated Gradients attribution.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    image : torch.Tensor
        Un-batched input image of shape ``(3, H, W)``.
    target : int
        Target class index for attribution.
    n_steps : int
        Number of interpolation steps. Default 50.
    device : str
        Device for computation.

    Returns
    -------
    torch.Tensor
        Signed attribution map of shape ``(3, H, W)`` on CPU.
    """
    image_batch = image.unsqueeze(0).to(device).requires_grad_(True)
    baseline = torch.zeros_like(image_batch)
    ig = IntegratedGradients(model)
    attr = ig.attribute(image_batch, baselines=baseline, target=target, n_steps=n_steps)
    return attr.squeeze(0).detach().cpu()


def compute_saliency(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute Saliency (absolute gradient) attribution.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    image : torch.Tensor
        Un-batched input image of shape ``(3, H, W)``.
    target : int
        Target class index for attribution.
    device : str
        Device for computation.

    Returns
    -------
    torch.Tensor
        Non-negative attribution map of shape ``(3, H, W)`` on CPU.
    """
    image_batch = image.unsqueeze(0).to(device).requires_grad_(True)
    sal = Saliency(model)
    attr = sal.attribute(image_batch, target=target, abs=True)
    return attr.squeeze(0).detach().cpu()


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    target_layer: nn.Module,
    device: str = "cpu",
    image_size: int = 224,
) -> torch.Tensor:
    """Compute GradCAM attribution via captum LayerGradCam.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    image : torch.Tensor
        Un-batched input image of shape ``(3, H, W)``.
    target : int
        Target class index for attribution.
    target_layer : nn.Module
        Convolutional layer to target (from ``get_gradcam_target_layer``).
    device : str
        Device for computation.
    image_size : int
        Spatial size for upsampling. Default 224.

    Returns
    -------
    torch.Tensor
        Non-negative attribution map of shape ``(3, H, W)`` on CPU.
        The single-channel GradCAM output is replicated across 3 channels
        to match the shape of gradient-based methods.
    """
    image_batch = image.unsqueeze(0).to(device).requires_grad_(True)
    gc = LayerGradCam(model, target_layer)
    attr = gc.attribute(image_batch, target=target)
    # Interpolate from low-resolution feature map to input size
    attr = LayerAttribution.interpolate(attr, (image_size, image_size), interpolate_mode="bilinear")
    # attr shape: (1, 1, H, W) — squeeze batch, replicate channel
    attr = attr.squeeze(0)  # (1, H, W)
    attr = attr.expand(3, -1, -1)  # (3, H, W)
    return attr.detach().cpu()


# Dispatch dict mapping method names to callables.
# Each value is a function (model, image, target, **kwargs) -> Tensor(3, H, W).
# GradCAM requires additional kwargs (target_layer, image_size).
FAE_METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "integrated_gradients": compute_integrated_gradients,
    "saliency": compute_saliency,
    "gradcam": compute_gradcam,
}


def compute_or_load(
    cache: Optional[AttributionCache],
    compute_fn: Callable[..., torch.Tensor],
    model_arch: str,
    fae_method: str,
    image_id: str,
    target: int,
    fae_hyperparams: dict,
    **compute_kwargs,
) -> torch.Tensor:
    """Return a cached attribution or compute, cache, and return it.

    Parameters
    ----------
    cache : AttributionCache or None
        If ``None``, caching is bypassed and *compute_fn* is always called.
    compute_fn : callable
        The attribution function to call on a cache miss.  Called as
        ``compute_fn(**compute_kwargs)``.
    model_arch : str
        Architecture name for the cache key.
    fae_method : str
        FAE method name for the cache key.
    image_id : str
        Image identifier for the cache key.
    target : int
        Target class index for the cache key.
    fae_hyperparams : dict
        Method hyperparameters whose hash guards against stale entries.
    **compute_kwargs
        Forwarded to *compute_fn* on a cache miss.

    Returns
    -------
    torch.Tensor
        Attribution map, typically shape ``(3, H, W)``.
    """
    if cache is not None:
        cached = cache.get(model_arch, fae_method, image_id, target, fae_hyperparams)
        if cached is not None:
            return cached

    # Ensure target is available to compute_fn even when stripped from
    # **compute_kwargs to avoid duplicate keyword arguments.
    if "target" not in compute_kwargs:
        compute_kwargs["target"] = target
    attr = compute_fn(**compute_kwargs)

    if cache is not None:
        cache.put(model_arch, fae_method, image_id, target, fae_hyperparams, attr)

    return attr
