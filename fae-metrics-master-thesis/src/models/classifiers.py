"""
Base classifier model definitions for ISIC 2017 (3-class skin lesion classification).

Responsibilities (see docs/thesis_plan.md §4, Models Under Evaluation):
    - Load ResNet-18 with final fc → Linear(512, num_classes)
    - Load SqueezeNet 1.1 with final classifier → Conv2d(512, num_classes, 1, 1)
    - Expose a helper to retrieve the GradCAM target layer per architecture

Not responsible for: training logic (see models/train.py),
or any ensemble/fusion architectures (those are in legacy/).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def load_resnet18(
    weights_path: str,
    num_classes: int = 3,
    device: str = "cpu",
) -> nn.Module:
    """Load a ResNet-18 classifier with pre-trained ISIC 2017 weights.

    Parameters
    ----------
    weights_path : str
        Path to the ``.pth`` state dict.
    num_classes : int
        Number of output classes. Default 3.
    device : str
        Target device. Default ``'cpu'``.

    Returns
    -------
    nn.Module
        The model in eval mode on the requested device.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_squeezenet(
    weights_path: str,
    num_classes: int = 3,
    device: str = "cpu",
) -> nn.Module:
    """Load a SqueezeNet 1.1 classifier with pre-trained ISIC 2017 weights.

    Parameters
    ----------
    weights_path : str
        Path to the ``.pth`` state dict.
    num_classes : int
        Number of output classes. Default 3.
    device : str
        Target device. Default ``'cpu'``.

    Returns
    -------
    nn.Module
        The model in eval mode on the requested device.
    """
    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
    model.num_classes = num_classes
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the layer to use as GradCAM target for a given architecture.

    Parameters
    ----------
    model : nn.Module
        The loaded model.
    arch : str
        Architecture identifier: ``'resnet18'`` or ``'squeezenet'``.

    Returns
    -------
    nn.Module
        The target layer for ``captum.attr.LayerGradCam``.

    Raises
    ------
    ValueError
        If *arch* is not recognised.

    Notes
    -----
    For SqueezeNet we target ``model.features[-1]`` (the last Fire module)
    rather than ``classifier.1`` (as used in the legacy code). The legacy
    choice pointed to the 1×1 classification convolution, which sits after
    the global average pool and produces a class-dimensional feature map
    with no spatial information — making GradCAM attribution trivial and
    uninformative. Targeting the last feature block preserves spatial
    resolution and yields meaningful saliency maps.
    """
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "squeezenet":
        return model.features[-1]
    raise ValueError(f"Unknown architecture '{arch}'. Expected 'resnet18' or 'squeezenet'.")
