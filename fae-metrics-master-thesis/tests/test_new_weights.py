"""Sanity-check: verify new ISIC 2017 weights load cleanly and produce
reasonable per-class predictions on the validation split.

Usage:
    python -m pytest tests/test_new_weights.py -v --tb=long
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.isic_dataset import CLASS_NAMES, ISIC2017Dataset
from src.models.classifiers import load_resnet18, load_squeezenet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "resnet18": {
        "loader": load_resnet18,
        "weights": "weights/resnet18_isic2017.pth",
    },
    "squeezenet": {
        "loader": load_squeezenet,
        "weights": "weights/squeezenet_isic2017.pth",
    },
}


@pytest.fixture(scope="module")
def val_dataset() -> ISIC2017Dataset:
    """Load the validation split once for the entire module."""
    ds = ISIC2017Dataset(root_dir="data", split="val", image_size=224, return_mask=False)
    assert len(ds) > 0, "Validation split is empty — check data/images/validation/"
    return ds


@pytest.fixture(scope="module", params=list(MODELS.keys()))
def loaded_model(request) -> tuple[str, torch.nn.Module]:
    """Load each model once per module run."""
    name = request.param
    cfg = MODELS[name]
    model = cfg["loader"](cfg["weights"], device=DEVICE)
    return name, model


class TestWeightsLoad:
    """STEP 4a-b: Verify state_dict loads cleanly."""

    @pytest.mark.parametrize("arch", list(MODELS.keys()))
    def test_state_dict_loads_cleanly(self, arch: str) -> None:
        cfg = MODELS[arch]
        model = cfg["loader"](cfg["weights"], device=DEVICE)
        # If we got here without RuntimeError, load was clean.
        # Explicitly verify model is in eval mode.
        assert not model.training, f"{arch} should be in eval mode after loading"


class TestPerClassAccuracy:
    """STEP 4c-e: Forward pass, per-class metrics, recall assertions."""

    @pytest.fixture(scope="class")
    def predictions(self, val_dataset: ISIC2017Dataset) -> dict[str, dict]:
        """Run forward pass for all models on the validation split.

        Returns dict: {arch_name: {"y_true": [...], "y_pred": [...]}}
        """
        results = {}
        for arch, cfg in MODELS.items():
            model = cfg["loader"](cfg["weights"], device=DEVICE)
            y_true, y_pred = [], []
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    sample = val_dataset[i]
                    img = sample["image"].unsqueeze(0).to(DEVICE)
                    logits = model(img)
                    pred = int(logits.argmax(dim=1).item())
                    y_true.append(sample["label"])
                    y_pred.append(pred)
            results[arch] = {"y_true": np.array(y_true), "y_pred": np.array(y_pred)}
        return results

    @pytest.mark.parametrize("arch", list(MODELS.keys()))
    def test_per_class_recall_above_threshold(
        self, arch: str, predictions: dict[str, dict]
    ) -> None:
        """Assert per-class recall > 0.40 for every class (catches majority collapse)."""
        data = predictions[arch]
        y_true, y_pred = data["y_true"], data["y_pred"]

        # Print full classification report for diagnostics
        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

        print(f"\n{'=' * 60}")
        print(f"MODEL: {arch}")
        print(f"{'=' * 60}")
        print(f"Validation images: {len(y_true)}")
        for i, cls in enumerate(CLASS_NAMES):
            count = int((y_true == i).sum())
            print(f"  {cls}: {count} samples")
        print(f"\nConfusion Matrix (rows=true, cols=pred):")
        print(f"{'':>25s} ", end="")
        for cls in CLASS_NAMES:
            print(f"{cls[:8]:>10s}", end="")
        print()
        for i, cls in enumerate(CLASS_NAMES):
            print(f"{cls:>25s} ", end="")
            for j in range(len(CLASS_NAMES)):
                print(f"{cm[i, j]:>10d}", end="")
            print()
        print(f"\nClassification Report:\n{report}")

        # Per-class recall check
        for i, cls in enumerate(CLASS_NAMES):
            n_true = int((y_true == i).sum())
            if n_true == 0:
                continue  # skip classes with no validation samples
            n_correct = int(cm[i, i])
            recall = n_correct / n_true
            assert recall >= 0.40, (
                f"{arch}: {cls} recall = {recall:.3f} (< 0.40). "
                f"Confusion matrix row: {cm[i, :].tolist()}. "
                f"Total samples for class: {n_true}"
            )

    @pytest.mark.parametrize("arch", list(MODELS.keys()))
    def test_overall_accuracy_above_50pct(
        self, arch: str, predictions: dict[str, dict]
    ) -> None:
        """Basic sanity: overall accuracy should exceed 50%."""
        data = predictions[arch]
        acc = (data["y_true"] == data["y_pred"]).mean()
        print(f"\n{arch} overall accuracy: {acc:.4f}")
        assert acc > 0.50, f"{arch} overall accuracy {acc:.4f} < 0.50"
