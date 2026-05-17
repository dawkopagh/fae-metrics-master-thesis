"""Phase 1 profiling: measure per-metric timing for one image across three FAE methods.

Outputs: results/profile_one_image.csv with columns:
    model, fae_method, image_id, metric, seconds
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.attributions.generate import (
    compute_gradcam,
    compute_occlusion,
    compute_saliency,
)
from src.data.isic_dataset import ISIC2017Dataset
from src.metrics.quantus_wrapper import compute_all_metrics
from src.models.classifiers import get_gradcam_target_layer, load_resnet18
from src.pipeline import _make_explain_func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROBE_FAE = ["saliency", "gradcam", "occlusion"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    model = load_resnet18("weights/resnet18_isic2017.pth", device=device)
    dataset = ISIC2017Dataset(root_dir="data", split="test", image_size=224, return_mask=True)
    sample = dataset[0]
    image_np = sample["image"].numpy()
    mask_np = sample["mask"].numpy() if sample["mask"] is not None else None
    image_id = sample["image_id"]

    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0).to(device))
        target = int(logits.argmax(dim=1).item())

    logger.info("Profiling image %s, target=%d", image_id, target)

    rows = []

    for fae_name in PROBE_FAE:
        logger.info("--- FAE: %s ---", fae_name)
        image_t = sample["image"]

        t_attr = time.perf_counter()
        if fae_name == "saliency":
            from src.attributions.generate import compute_saliency
            attr_t = compute_saliency(model, image_t, target, device=device)
        elif fae_name == "gradcam":
            tl = get_gradcam_target_layer(model, "resnet18")
            attr_t = compute_gradcam(model, image_t, target, tl, device=device)
        elif fae_name == "occlusion":
            attr_t = compute_occlusion(model, image_t, target, device=device,
                                       sliding_window_shapes=(3, 15, 15),
                                       strides=(3, 8, 8))
        attr_np = attr_t.numpy()
        logger.info("  attribution: %.2fs", time.perf_counter() - t_attr)

        explain_func = _make_explain_func(fae_name, model, "resnet18", device)

        _, timings = compute_all_metrics(
            model=model,
            image=image_np,
            attribution=attr_np,
            target=target,
            mask=mask_np,
            device=device,
            explain_func=explain_func,
            fae_method=fae_name,
            return_timings=True,
        )

        for metric, secs in timings.items():
            rows.append({
                "model": "resnet18",
                "fae_method": fae_name,
                "image_id": image_id,
                "metric": metric,
                "seconds": round(secs, 3),
            })
            logger.info("  %-35s %.2fs", metric, secs)

    df = pd.DataFrame(rows)
    out = Path("results/profile_one_image.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved to %s", out)

    # --- summary sorted by seconds desc ---
    print("\n=== PER-METRIC TIMING SUMMARY (sorted by avg seconds desc) ===")
    summary = (
        df.groupby(["fae_method", "metric"])["seconds"]
        .sum()
        .reset_index()
        .sort_values("seconds", ascending=False)
    )
    print(summary.to_string(index=False))

    slow = df[df["seconds"] > 5]
    if not slow.empty:
        print(f"\n>>> SLOW (>5s): {len(slow)} combinations <<<")
        print(slow.to_string(index=False))


if __name__ == "__main__":
    main()
