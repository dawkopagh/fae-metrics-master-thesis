"""
PyTorch Dataset and DataLoader wrappers for ISIC 2017 (Task 1 — Lesion Segmentation).

Responsibilities (see docs/thesis_plan.md §6, Data):
    - Load images and paired segmentation masks from the ISIC 2017 directory layout
      (data/{images,masks}/{train,validation,test}/{class}/)
    - Return dicts with image tensor, mask tensor, class index, image id
    - Apply ImageNet normalization and mask binarization

Not responsible for: downloading the dataset (see download_isic.py),
attribution generation (see attributions/generate.py), or
attribution caching (see attributions/cache.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Alphabetical order — matches torchvision.datasets.ImageFolder convention
# used in train_base_classifiers.ipynb.
CLASS_NAMES: list[str] = ["melanoma", "nevus", "seborrheic_keratosis"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_SPLIT_DIR_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


class ISIC2017Dataset(Dataset):
    """ISIC 2017 skin lesion dataset with paired segmentation masks.

    Parameters
    ----------
    root_dir : str
        Path to the top-level data directory containing ``images/`` and
        ``masks/`` sub-trees.
    split : str
        One of ``'train'``, ``'val'``, or ``'test'``.
    image_size : int
        Target spatial resolution (square). Default 224.
    return_mask : bool
        If True (default), load the paired segmentation mask. When a mask
        file is missing, the ``'mask'`` value will be ``None``.

    Returns
    -------
    dict
        ``{'image': Tensor(3, H, W), 'mask': Tensor(1, H, W) | None,
        'label': int, 'image_id': str}``
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        image_size: int = 224,
        return_mask: bool = True,
    ) -> None:
        if split not in _SPLIT_DIR_MAP:
            raise ValueError(
                f"split must be one of {list(_SPLIT_DIR_MAP.keys())}, got '{split}'"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.return_mask = return_mask

        split_dir = _SPLIT_DIR_MAP[split]
        self.image_root = self.root_dir / "images" / split_dir
        self.mask_root = self.root_dir / "masks" / split_dir

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )

        self.samples: list[tuple[str, int, str]] = []  # (image_path, label, image_id)
        self._build_sample_list()

    def _build_sample_list(self) -> None:
        """Scan the class sub-directories and build the sample list."""
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.image_root / class_name
            if not class_dir.is_dir():
                continue
            for fname in sorted(os.listdir(class_dir)):
                # Skip macOS resource-fork metadata files
                if fname.startswith("._"):
                    continue
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                image_id = Path(fname).stem  # e.g. "ISIC_0012258"
                image_path = str(class_dir / fname)
                self.samples.append((image_path, label_idx, image_id))

    def _find_mask_path(self, image_id: str, label_idx: int) -> Optional[str]:
        """Locate the segmentation mask for a given image id."""
        class_name = CLASS_NAMES[label_idx]
        mask_dir = self.mask_root / class_name
        mask_name = f"{image_id}_segmentation.png"
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            return str(mask_path)
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, label_idx, image_id = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        mask_tensor: Optional[torch.Tensor] = None
        if self.return_mask:
            mask_path = self._find_mask_path(image_id, label_idx)
            if mask_path is not None:
                mask = Image.open(mask_path).convert("L")
                mask_tensor = self.mask_transform(mask)
                # Binarize: > 0.5 → 1.0
                mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_idx,
            "image_id": image_id,
        }
