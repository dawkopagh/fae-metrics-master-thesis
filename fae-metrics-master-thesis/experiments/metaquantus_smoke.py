"""Single-fixture integration smoke test for MetaQuantus NR + AR tests.

Scope: 1 image × 1 model (ResNet-18) × 1 FAE (Integrated Gradients)
       × 1 metric (FaithfulnessCorrelation).

Validates that the full integration path
  load model → compute attribution → meta_evaluate_metric → print scores
works end-to-end on CPU before scaling on Colab.

FaithfulnessCorrelation is configured with reduced nr_runs (10 instead of 100)
to stay within the 90-second wall-clock budget on CPU.  The default production
configuration (nr_runs=100) is used on Colab.

Expected output (one line):
  FaithfulnessCorrelation: NR=0.xx, AR=0.xx, combined=0.xx

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
        # inputs: (B, C, H, W) float32 numpy; targets: (B,) int numpy
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

    # --- synthetic test image (no data dependency) ---
    rng = np.random.default_rng(0)
    image_np = rng.random((3, 224, 224)).astype(np.float32)
    image_t = torch.tensor(image_np, dtype=torch.float32)

    # --- predict target class ---
    with torch.no_grad():
        logits = model(image_t.unsqueeze(0).to(device))
        target = int(logits.argmax(dim=1).item())

    # --- compute baseline attribution ---
    attr_t = compute_integrated_gradients(model, image_t, target, device=device, n_steps=20)
    attr_np = attr_t.numpy()

    # --- configure FaithfulnessCorrelation with reduced cost for CPU ---
    # nr_runs=10 (vs 100 in production) cuts runtime ~10x on CPU.
    fc_metric = quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=224,
        perturb_baseline="black",
        normalise=True,
        abs=False,
        return_aggregate=False,
        disable_warnings=True,
    )

    explain_fn = _make_ig_explain_fn(device)

    # --- meta-evaluation: NR (n_seeds=3) + AR (n_levels=3) ---
    result = meta_evaluate_metric(
        metric_name="FaithfulnessCorrelation",
        metric_fn=fc_metric,
        model=model,
        images=[image_np],
        attributions=[attr_np],
        targets=[target],
        explain_fn=explain_fn,
        device=device,
        n_seeds=3,
        n_levels=3,
    )

    elapsed = time.perf_counter() - t_start

    nr = result["nr"]["nr_score"]
    ar = result["ar"]["ar_score"]
    combined = result["combined_reliability"]

    print(
        f"FaithfulnessCorrelation: NR={nr:.2f}, AR={ar:.2f}, combined={combined:.2f}"
    )
    print(f"Wall clock: {elapsed:.1f}s")

    # --- assertions ---
    assert elapsed < WALL_CLOCK_LIMIT_S, (
        f"Smoke test exceeded {WALL_CLOCK_LIMIT_S}s wall-clock limit: {elapsed:.1f}s\n"
        f"Bottleneck: FaithfulnessCorrelation calls {3 + 3} times "
        f"(3 NR seeds + 3 AR levels), each runs 10 internal perturbation rounds. "
        f"On CPU with IG attributions (20 steps), each meta-call takes "
        f"~{elapsed / 6:.1f}s.  Switch to Colab T4 for the full run."
    )

    assert np.isfinite(nr), f"nr_score is not finite: {nr}"
    assert np.isfinite(ar), f"ar_score is not finite: {ar}"
    assert np.isfinite(combined), f"combined_reliability is not finite: {combined}"
    assert 0.0 <= nr <= 1.0, f"nr_score out of [0, 1]: {nr}"
    assert 0.0 <= ar <= 1.0, f"ar_score out of [0, 1]: {ar}"


if __name__ == "__main__":
    main()
