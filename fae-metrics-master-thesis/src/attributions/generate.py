"""
Attribution map generation for a configurable set of FAE methods via captum.

Responsibilities (see docs/thesis_plan.md §5, FAE Methods Evaluated):
    - Compute per-image attribution maps for Integrated Gradients, Saliency,
      GradCAM, DeepLift, GuidedBackprop, LRP, and Occlusion
    - Return attributions in canonical shape (3, H, W) on CPU
    - Provide a FAE_METHODS dispatch dict for pipeline iteration

Not responsible for: caching attributions to disk (see cache.py),
ensembling multiple attribution maps (see ensembling.py),
or metric computation (see metrics/).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from captum.attr import (
    DeepLift,
    GuidedBackprop,
    IntegratedGradients,
    LayerAttribution,
    LayerGradCam,
    LRP,
    Occlusion,
    Saliency,
)

from src.attributions.cache import AttributionCache

logger = logging.getLogger(__name__)


def _disable_inplace_relu(model: nn.Module) -> list[nn.ReLU]:
    """Temporarily set ``inplace=False`` on all ReLU modules.

    Returns the list of modules that were modified so the caller can
    restore them via :func:`_restore_inplace_relu`.  Required by
    DeepLift and LRP which register hooks incompatible with in-place ops.
    """
    modified: list[nn.ReLU] = []
    for m in model.modules():
        if isinstance(m, nn.ReLU) and m.inplace:
            m.inplace = False
            modified.append(m)
    return modified


def _restore_inplace_relu(modules: list[nn.ReLU]) -> None:
    """Restore ``inplace=True`` on previously modified ReLU modules."""
    for m in modules:
        m.inplace = True


def _replace_shared_relus(model: nn.Module) -> None:
    """Replace ReLU modules reused in ``forward()`` with unique instances.

    torchvision's ``BasicBlock`` (ResNet) and ``Fire`` (SqueezeNet) reuse
    a single ``self.relu`` for multiple call sites.  DeepLift requires each
    module to be invoked exactly once.  This function replaces each ``relu``
    attribute that is a ``nn.ReLU`` with a fresh ``nn.ReLU(inplace=False)``
    and additionally creates a ``relu2`` attribute for ``BasicBlock`` which
    calls ``self.relu`` twice in ``forward()``.
    """
    from torchvision.models.resnet import BasicBlock

    for name, module in model.named_modules():
        if isinstance(module, BasicBlock):
            # BasicBlock uses self.relu in two places; replace with two
            module.relu = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            # Monkey-patch forward to use relu2 for the second call
            _patch_basicblock_forward(module)
        elif hasattr(module, "relu") and isinstance(module.relu, nn.ReLU):
            module.relu = nn.ReLU(inplace=False)


def _patch_basicblock_forward(block: nn.Module) -> None:
    """Monkey-patch a BasicBlock's forward to use ``relu`` and ``relu2``."""
    import torch
    from torchvision.models.resnet import BasicBlock

    def _forward(self: BasicBlock, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)  # second ReLU uses a separate module
        return out

    import types
    block.forward = types.MethodType(_forward, block)


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


def compute_deep_lift(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute DeepLift attribution.

    Uses a zero baseline (default). DeepLift propagates activation
    differences with respect to the baseline through the network using
    modified chain rules.

    A deep copy of the model is created internally to replace shared
    ReLU modules with unique instances (required by captum's DeepLift)
    and to disable in-place operations.

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
        Signed attribution map of shape ``(3, H, W)`` on CPU.
    """
    import copy
    model_copy = copy.deepcopy(model)
    _replace_shared_relus(model_copy)
    _disable_inplace_relu(model_copy)
    model_copy.eval()

    image_batch = image.unsqueeze(0).to(device).requires_grad_(True)
    baseline = torch.zeros_like(image_batch)
    dl = DeepLift(model_copy)
    attr = dl.attribute(image_batch, baselines=baseline, target=target)
    return attr.squeeze(0).detach().cpu()


def compute_guided_backprop(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute Guided Backpropagation attribution.

    GuidedBackprop registers forward/backward hooks that modify gradient
    flow through ReLU layers (clamping negative gradients). It is NOT
    safe inside a ``torch.no_grad()`` context — gradients must be enabled.
    The model should be in eval mode (batch-norm frozen).

    Parameters
    ----------
    model : nn.Module
        Classifier in **eval mode**. Must not be wrapped in
        ``torch.no_grad()``.
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
    modified = _disable_inplace_relu(model)
    try:
        with torch.enable_grad():
            image_batch = image.unsqueeze(0).to(device).requires_grad_(True)
            gbp = GuidedBackprop(model)
            attr = gbp.attribute(image_batch, target=target)
    finally:
        _restore_inplace_relu(modified)
    return attr.squeeze(0).detach().cpu()


def compute_lrp(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute Layer-wise Relevance Propagation (LRP) attribution.

    Uses captum's default LRP rules. These work for standard
    architectures (ResNet-18, SqueezeNet 1.1). If the model contains
    unsupported layer types, a ``RuntimeError`` is raised with a
    message identifying the problematic layer.

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
        Signed attribution map of shape ``(3, H, W)`` on CPU.

    Raises
    ------
    RuntimeError
        If a layer type is not supported by captum's default LRP rules.
    """
    modified = _disable_inplace_relu(model)
    image_batch = image.unsqueeze(0).to(device).requires_grad_(True)
    try:
        lrp = LRP(model)
        attr = lrp.attribute(image_batch, target=target)
    except Exception as exc:
        raise RuntimeError(
            f"LRP failed on model {type(model).__name__}. This typically "
            f"indicates an unsupported layer type. Original error: {exc}"
        ) from exc
    finally:
        _restore_inplace_relu(modified)
    return attr.squeeze(0).detach().cpu()


def compute_occlusion(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    device: str = "cpu",
    sliding_window_shapes: Tuple[int, int, int] = (3, 15, 15),
    strides: Tuple[int, int, int] = (3, 8, 8),
) -> torch.Tensor:
    """Compute Occlusion-based attribution.

    Slides a window across the input, replacing each patch with a
    baseline value (zero) and measuring the change in the model's
    output. This is a perturbation-based (black-box) method.

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
    sliding_window_shapes : tuple of int
        Shape of the sliding window ``(C, H, W)``.
        Default ``(3, 15, 15)``.
    strides : tuple of int
        Stride of the sliding window ``(C, H, W)``.
        Default ``(3, 8, 8)``.

    Returns
    -------
    torch.Tensor
        Signed attribution map of shape ``(3, H, W)`` on CPU.
    """
    image_batch = image.unsqueeze(0).to(device)
    occ = Occlusion(model)
    attr = occ.attribute(
        image_batch,
        target=target,
        sliding_window_shapes=sliding_window_shapes,
        strides=strides,
        baselines=0,
    )
    return attr.squeeze(0).detach().cpu()


# Dispatch dict mapping method names to callables.
# Each value is a function (model, image, target, **kwargs) -> Tensor(3, H, W).
# GradCAM requires additional kwargs (target_layer, image_size).
# Occlusion accepts optional sliding_window_shapes and strides kwargs.
FAE_METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "integrated_gradients": compute_integrated_gradients,
    "saliency": compute_saliency,
    "gradcam": compute_gradcam,
    "deep_lift": compute_deep_lift,
    "guided_backprop": compute_guided_backprop,
    "lrp": compute_lrp,
    "occlusion": compute_occlusion,
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
