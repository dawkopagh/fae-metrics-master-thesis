"""Single-fixture integration smoke test for MetaQuantus NR + AR tests.

Scope: 1 image × 1 model (ResNet-18) × 3 metrics:
  - FaithfulnessCorrelation  (no auxiliary inputs)
  - PointingGame             (needs mask)
  - MaxSensitivity           (needs explain_func)

Validates that the full integration path works end-to-end on CPU, including
correct forwarding of masks and explain_func, before scaling on Colab.

FaithfulnessCorrelation and MaxSensitivity use reduced nr_runs / nr_samples
to stay within the 90-second wall-clock budget on CPU.

Expected output (three lines):
  FaithfulnessCorrelation: NR=0.xx, AR=0.xx, combined=0.xx
  PointingGame:            NR=0.xx, AR=0.xx, combined=0.xx
  MaxSensitivity:          NR=0.xx, AR=0.xx, combined=0.xx

Run from the project root:
  python experiments/metaquantus_smoke.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import quantus
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.attributions.generate import compute_integrated_gradients
from src.meta_evaluation.metaquantus_wrapper import meta_evaluate_metric
from src.models.classifiers import load_resnet18

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

WALL_CLOCK_LIMIT_S = 90.0


def _make_ig_explain_fn(device: str):
    """Build an explain_fn for Integrated Gradients with Quantus signature."""

    def _explain(model, inputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
        attrs = []
        for i in range(inputs.shape[0]):
            img_t = torch.tensor(inputs[i], dtype=torch.float32)
            tgt = int(targets[i])
            attr_t = compute_integrated_gradients(
                model, img_t, tgt, device=device, n_steps=20
            )
            attrs.append(attr_t.numpy())
        return np.stack(attrs, axis=0)

    return _explain


def _run_metric(name, metric_fn, model, image_np, attr_np, target,
                explain_fn, mask_np, device):
    result = meta_evaluate_metric(
        metric_name=name,
        metric_fn=metric_fn,
        model=model,
        images=[image_np],
        attributions=[attr_np],
        targets=[target],
        explain_fn=explain_fn,
        masks=[mask_np] if mask_np is not None else None,
        device=device,
        n_seeds=3,
        n_levels=3,
    )
    nr = result["nr"]["nr_score"]
    ar = result["ar"]["ar_score"]
    combined = result["combined_reliability"]
    nr_str = f"{nr:.2f}" if np.isfinite(nr) else "NaN"
    ar_str = f"{ar:.2f}" if np.isfinite(ar) else "NaN"
    combined_str = f"{combined:.2f}" if np.isfinite(combined) else "NaN"
    print(f"{name:30s}: NR={nr_str}, AR={ar_str}, combined={combined_str}")
    return nr, ar, combined


def main() -> None:
    t_start = time.perf_counter()
    device = "cpu"

    # --- load model ---
    weights_path = _PROJECT_ROOT / "weights" / "resnet18_isic2017.pth"
    if not weights_path.exists():
        sys.exit(
            f"Weights not found: {weights_path}\n"
            "Download or train ResNet-18 before running the smoke test."
        )
    model = load_resnet18(str(weights_path), device=device)

    # --- synthetic test image and mask (no data dependency) ---
    rng = np.random.default_rng(0)
    image_np = rng.random((3, 224, 224)).astype(np.float32)
    image_t = torch.tensor(image_np, dtype=torch.float32)

    # Synthetic mask: center 112×112 square (simulates a lesion region)
    mask_np = np.zeros((1, 224, 224), dtype=np.float32)
    mask_np[0, 56:168, 56:168] = 1.0

    # --- predict target class ---
    with torch.no_grad():
        logits = model(image_t.unsqueeze(0).to(device))
        target = int(logits.argmax(dim=1).item())

    # --- compute baseline attribution ---
    attr_t = compute_integrated_gradients(model, image_t, target, device=device, n_steps=20)
    attr_np = attr_t.numpy()

    explain_fn = _make_ig_explain_fn(device)

    # --- 1. FaithfulnessCorrelation (no auxiliary inputs) ---
    fc_metric = quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=224,
        perturb_baseline="black",
        normalise=True,
        abs=False,
        return_aggregate=False,
        disable_warnings=True,
    )
    nr_fc, ar_fc, combined_fc = _run_metric(
        "FaithfulnessCorrelation", fc_metric,
        model, image_np, attr_np, target, explain_fn, None, device,
    )

    # --- 2. PointingGame (needs mask) ---
    pg_metric = quantus.PointingGame(
        normalise=True,
        abs=True,
        return_aggregate=False,
        disable_warnings=True,
    )
    nr_pg, ar_pg, combined_pg = _run_metric(
        "PointingGame", pg_metric,
        model, image_np, attr_np, target, explain_fn, mask_np, device,
    )

    # --- 3. MaxSensitivity (needs explain_func) ---
    ms_metric = quantus.MaxSensitivity(
        nr_samples=5,
        lower_bound=0.2,
        normalise=False,
        abs=False,
        return_aggregate=False,
        disable_warnings=True,
    )
    nr_ms, ar_ms, combined_ms = _run_metric(
        "MaxSensitivity", ms_metric,
        model, image_np, attr_np, target, explain_fn, None, device,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nWall clock: {elapsed:.1f}s")

    # --- assertions ---
    assert elapsed < WALL_CLOCK_LIMIT_S, (
        f"Smoke test exceeded {WALL_CLOCK_LIMIT_S}s wall-clock limit: {elapsed:.1f}s"
    )
    for name, nr, ar, combined in [
        ("FaithfulnessCorrelation", nr_fc, ar_fc, combined_fc),
        ("PointingGame", nr_pg, ar_pg, combined_pg),
        ("MaxSensitivity", nr_ms, ar_ms, combined_ms),
    ]:
        assert np.isfinite(nr), f"{name}: nr_score not finite: {nr}"
        assert np.isfinite(ar), f"{name}: ar_score not finite: {ar}"
        assert np.isfinite(combined), f"{name}: combined_reliability not finite: {combined}"
        assert 0.0 <= nr <= 1.0, f"{name}: nr_score={nr} out of [0, 1]"
        assert 0.0 <= ar <= 1.0, f"{name}: ar_score={ar} out of [0, 1]"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
