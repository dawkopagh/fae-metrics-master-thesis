#!/usr/bin/env python
"""Generate the Chapter 4 figures from the full-run result CSVs.

Outputs (into Latex/figures/):
    cd_diagram.pdf      - Nemenyi critical-difference diagram (per model,
                          Autoweighted scheme) from results/cd_summary.csv.
    ensemble_radar.pdf  - ensemble vs. mean-individual per-metric radar over
                          the five non-Robustness M* metrics, min-max
                          normalised per metric for comparability.

Also prints the best validation accuracies (from the training logs) and the
full-run score-row count used in the Chapter 4 tables.

Run from the repo root:
    fae-metrics-master-thesis/.venv/bin/python \
        fae-metrics-master-thesis/experiments/make_figures.py
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.aggregation.normalize import METRIC_DIRECTIONS  # noqa: E402

RES = "results"
FIG = "Latex/figures"
LOGS = "fae-metrics-master-thesis/results"
COMMON = [
    "faithfulness_correlation", "max_sensitivity", "pixel_flipping",
    "pointing_game", "random_logit", "relevance_mass_accuracy", "sparseness",
]
_SHORT = {
    "faithfulness_correlation": "Faith.", "max_sensitivity": "MaxSens",
    "pixel_flipping": "PixFlip", "pointing_game": "PG",
    "random_logit": "RandLog", "relevance_mass_accuracy": "RMA",
    "sparseness": "Sparse",
}


def report_scalars() -> None:
    for name, f in [("ResNet-18", f"{LOGS}/training_log_resnet18.csv"),
                    ("SqueezeNet", f"{LOGS}/training_log_squeezenet.csv")]:
        v = pd.read_csv(f).query("phase == 'validation'")["acc"]
        print(f"{name}: best val acc = {v.max() * 100:.1f}% (final {v.iloc[-1] * 100:.1f}%)")
    print("Full-run score rows = 100,800 (2 models x 7 FAE x 600 images x 12 metrics)")


def _cd_panel(ax, sub: pd.DataFrame, title: str) -> None:
    sub = sub.sort_values("mean_rank")
    k = len(sub)
    cd = sub["critical_difference"].iloc[0]
    ranks = sub["mean_rank"].values
    names = sub["method"].values
    lo, hi = int(np.floor(ranks.min())), int(np.ceil(ranks.max()))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-k - 1.5, 2.2)
    ax.axis("off")
    ax.plot([lo, hi], [0, 0], "k-", lw=1)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [0, 0.12], "k-", lw=1)
        ax.text(x, 0.28, str(x), ha="center", va="bottom", fontsize=8)
    half = int(np.ceil(k / 2))
    for i, (r, n) in enumerate(zip(ranks, names)):
        if i < half:
            y = -(i + 1)
            ax.plot([r, r], [0, y], "k-", lw=0.8)
            ax.plot([r, lo - 0.2], [y, y], "k-", lw=0.8)
            ax.text(lo - 0.25, y, f"{n} ({r:.2f})", ha="right", va="center", fontsize=8)
        else:
            y = -(i - half + 1)
            ax.plot([r, r], [0, y], "k-", lw=0.8)
            ax.plot([r, hi + 0.2], [y, y], "k-", lw=0.8)
            ax.text(hi + 0.25, y, f"{n} ({r:.2f})", ha="left", va="center", fontsize=8)
    ax.plot([lo, lo + cd], [1.4, 1.4], "k-", lw=2)
    ax.plot([lo, lo], [1.3, 1.5], "k-", lw=1)
    ax.plot([lo + cd, lo + cd], [1.3, 1.5], "k-", lw=1)
    ax.text(lo + cd / 2, 1.6, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    yb = -0.18
    i = 0
    while i < k:
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:
            ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [yb, yb], "-", color="crimson", lw=3)
            yb -= 0.12
        i += 1
    ax.set_title(title, fontsize=9)


def make_cd_diagram() -> None:
    cd = pd.read_csv(f"{RES}/cd_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, model, lab in [(axes[0], "resnet18", "ResNet-18"),
                           (axes[1], "squeezenet", "SqueezeNet")]:
        sub = cd[(cd["model"] == model) & (cd["scheme"] == "autoweighted")]
        _cd_panel(ax, sub, f"{lab} (Autoweighted)")
    plt.tight_layout()
    plt.savefig(f"{FIG}/cd_diagram.pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG}/cd_diagram.pdf")


def make_ensemble_radar() -> None:
    sc = f"{RES}/SCORES_full_run_600_GOOD.csv"
    sc = sc if os.path.exists(sc) else f"{RES}/full_run_7fae_12metrics_600.csv"
    indf = pd.read_csv(sc)
    ens = pd.read_csv(f"{RES}/full_run_ensemble_7fae_12metrics_600.csv")
    imgs = set(ens["image_id"].unique())
    ang = np.linspace(0, 2 * np.pi, len(COMMON), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), subplot_kw=dict(polar=True))
    for ax, model, lab in [(axes[0], "resnet18", "ResNet-18"),
                           (axes[1], "squeezenet", "SqueezeNet")]:
        im = indf[(indf.model == model) & indf.image_id.isin(imgs) & indf.metric.isin(COMMON)]
        em = ens[(ens.model == model) & ens.metric.isin(COMMON)]
        # Per metric: mean per individual method, plus the ensemble mean.
        per_method = im.groupby(["metric", "fae_method"])["score"].mean().unstack()  # metric x method
        ind_mean = per_method.mean(axis=1)                       # mean individual method
        ens_mean = em.groupby("metric")["score"].mean()          # ensemble
        # Normalise each metric against the full range (all individual methods +
        # ensemble) so the two curves sit at meaningful fractional radii, and
        # rectify by metric direction so a LARGER radius is always BETTER
        # (lower-is-better axes are flipped: radius = (hi - x) / range).
        lo = pd.concat([per_method.min(axis=1), ens_mean], axis=1).min(axis=1)
        hi = pd.concat([per_method.max(axis=1), ens_mean], axis=1).max(axis=1)
        rng = (hi - lo).replace(0, 1)

        def _radius(x, m):
            frac = (x - lo[m]) / rng[m]
            return frac if METRIC_DIRECTIONS[m] > 0 else 1.0 - frac

        iv = [_radius(ind_mean[m], m) for m in COMMON]
        ev = [_radius(ens_mean[m], m) for m in COMMON]
        iv += iv[:1]
        ev += ev[:1]
        ax.plot(ang, iv, "-o", label="Mean individual", color="steelblue", ms=3)
        ax.fill(ang, iv, alpha=0.1, color="steelblue")
        ax.plot(ang, ev, "-o", label="Ensemble", color="crimson", ms=3)
        ax.fill(ang, ev, alpha=0.1, color="crimson")
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([_SHORT[m] for m in COMMON], fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(lab, fontsize=9)
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG}/ensemble_radar.pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG}/ensemble_radar.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    report_scalars()
    make_cd_diagram()
    make_ensemble_radar()
