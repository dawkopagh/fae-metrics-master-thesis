"""
Training loop for the base classifiers on ISIC 2017.

Trains ResNet-18 and SqueezeNet 1.1 with class-weighted cross-entropy,
SGD + momentum, and StepLR scheduling. Saves best-validation-accuracy
checkpoint to disk.

Not responsible for: model architecture definitions (see classifiers.py),
attribution generation (see attributions/generate.py),
or any evaluation metrics (see metrics/).
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[dict[str, DataLoader], dict[str, int], list[str], list[int]]:
    """Build train/validation/test DataLoaders via ImageFolder.

    Parameters
    ----------
    data_root : str
        Root containing ``images/{train,validation,test}/{class}/``.
    image_size : int
        Target spatial size for resize.
    batch_size : int
        Batch size.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    tuple
        (dataloaders dict, dataset_sizes dict, class_names list, train_labels list)
    """
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }

    images_root = os.path.join(data_root, "images")

    def _is_valid_file(path: str) -> bool:
        """Filter out macOS resource fork files (._*)."""
        return not os.path.basename(path).startswith("._")

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(images_root, x),
            data_transforms[x],
            is_valid_file=_is_valid_file,
        )
        for x in ["train", "validation", "test"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=num_workers,
        )
        for x in ["train", "validation", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation", "test"]}
    class_names = image_datasets["train"].classes
    train_labels = [s[1] for s in image_datasets["train"].samples]

    return dataloaders, dataset_sizes, class_names, train_labels


def _create_model(arch: str, num_classes: int = 3) -> nn.Module:
    """Instantiate a model with ImageNet-pretrained backbone.

    Parameters
    ----------
    arch : str
        ``'resnet18'`` or ``'squeezenet'``.
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
    """
    if arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "squeezenet":
        model = models.squeezenet1_1(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        model.num_classes = num_classes
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def train_classifier(
    arch: str,
    data_root: str,
    output_weights_path: str,
    device: str = "cpu",
    epochs: int = 25,
    lr: float = 0.001,
    momentum: float = 0.9,
    step_size: int = 7,
    gamma: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Train a classifier on ISIC 2017 and save the best checkpoint.

    Parameters
    ----------
    arch : str
        Architecture: ``'resnet18'`` or ``'squeezenet'``.
    data_root : str
        Root data directory (containing ``images/{train,validation,test}/``).
    output_weights_path : str
        Where to save the best state dict.
    device : str
        Torch device string.
    epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate for SGD.
    momentum : float
        SGD momentum.
    step_size : int
        ``StepLR`` step size (epochs).
    gamma : float
        ``StepLR`` decay factor.
    batch_size : int
        Batch size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Per-epoch log with columns: ``epoch, phase, loss, acc``.
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Training %s | data=%s | device=%s | epochs=%d | seed=%d",
                arch, data_root, device, epochs, seed)

    dataloaders, dataset_sizes, class_names, train_labels = _build_dataloaders(
        data_root, batch_size=batch_size
    )
    logger.info("Classes: %s", class_names)
    logger.info("Dataset sizes: %s", dataset_sizes)

    # Class weights for imbalanced data
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=np.array(train_labels))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info("Class weights: %s", dict(zip(class_names, weights.tolist())))

    model = _create_model(arch, num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log_rows: list[dict] = []

    t0 = time.time()
    for epoch in range(epochs):
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            log_rows.append({
                "epoch": epoch + 1,
                "phase": phase,
                "loss": epoch_loss,
                "acc": epoch_acc,
            })

            logger.info(
                "Epoch %d/%d [%s] Loss: %.4f Acc: %.4f",
                epoch + 1, epochs, phase, epoch_loss, epoch_acc,
            )

            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0
    logger.info("Training complete: %.1f s | Best val acc: %.4f", elapsed, best_acc)

    # Save best weights
    output_path = Path(output_weights_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), output_path)
    logger.info("Saved weights → %s", output_path)

    return pd.DataFrame(log_rows)
