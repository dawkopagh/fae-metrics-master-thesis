#!/usr/bin/env python
"""Generate the Chapter 4 figures from the full-run result CSVs.

Outputs (into Latex/figures/):
    cd_diagram.pdf      - Nemenyi critical-difference diagram (per model,
                          Autoweighted scheme) from results/cd_summary.csv.
    ensemble_radar.pdf  - ensemble vs. mean-individual per-metric radar over
                          all seven M* metrics, min-max
                          normalised per metric for comparability.
    reliability_bars.pdf - NR and AR per M* metric (mean over the 14 cells,
                          SD whiskers) against each metric's own AR noise
                          floor, from results/meta_evaluation_reliability.csv.
    redundancy_bars.pdf - all 36 pairwise Spearman |rho| per model, sorted,
                          against the |rho| > 0.85 pruning threshold, from
                          results/redundancy_matrix_*.csv.

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


def make_ar_schematic() -> None:
    """AR-estimator illustration for Chapter 3, from measured curves.

    Two real (model, FAE, metric) cells from the extended-AR run: one
    reacting in the expected direction (both plotted metrics are
    lower-is-better, so degradation should RAISE the score) and one
    reacting the unexpected way, which the direction-agnostic
    AR = |Spearman rho| nonetheless credits. Prefers the v2 (n=64,
    10-level) artifact when present; falls back to the v1 completion CSV.
    """
    v2 = Path(f"{RES}/ar_completion_v2_n64.csv")
    src = v2 if v2.exists() else Path(f"{RES}/ar_completion_progress.csv")
    ar = pd.read_csv(src)

    cells = [
        ("resnet18", "integrated_gradients", "max_sensitivity",
         "steelblue", "expected direction"),
        ("resnet18", "guided_backprop", "random_logit",
         "crimson", "unexpected direction"),
    ]
    _FAE_LABEL = {"integrated_gradients": "Integrated Gradients",
                  "guided_backprop": "Guided Backprop"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, (model, fae, metric, color, tag) in zip(axes, cells):
        sel = ar[(ar.model == model) & (ar.fae_method == fae)
                 & (ar.metric == metric)]
        if sel.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"({model}, {fae}, {metric})\nnot in {src.name}",
                    ha="center", va="center", fontsize=8)
            continue
        row = sel.iloc[0]
        scores = [float(s) for s in str(row.scores).split(";")]
        levels = np.linspace(0.0, 0.9, len(scores))
        rho = float(row.monotonicity)

        ax.plot(levels, scores, "-o", color=color, lw=1.5, ms=4)
        ax.set_title(f"{_SHORT.get(metric, metric)} / {_FAE_LABEL[fae]} "
                     f"(ResNet-18)", fontsize=9)
        ax.set_xlabel("degradation fraction $f$", fontsize=8)
        ax.set_ylabel("mean metric score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        # Annotate in whichever top corner is free of the curve's start.
        left_corner = scores[0] < max(scores)
        ax.text(0.03 if left_corner else 0.97, 0.95,
                f"{tag}\n$\\rho_S = {rho:+.1f}$"
                f"$\\;\\Rightarrow\\;$AR$\\,= {abs(rho):.1f}$",
                transform=ax.transAxes,
                ha="left" if left_corner else "right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                          lw=0.5))
    plt.tight_layout()
    plt.savefig(f"{FIG}/ar_schematic.pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG}/ar_schematic.pdf (source: {src.name})")



# Colours: dataviz reference palette, categorical slots 1-2 (a validated
# adjacent pair on a light surface); text and grid never wear the data colour.
_BLUE, _ORANGE = "#2a78d6", "#eb6834"
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#dcdbd7"
_SHORT_ALL = {**_SHORT, "avg_sensitivity": "AvgSens", "complexity": "Complex"}
# Expected |rho_S| of a random ordering of the degradation levels: exactly
# 0.42 for the five-level protocol (mean over all 120 permutations), ~0.27
# for the ten-level extended-AR protocol of the two explain_func metrics.
_AR_FLOOR = {m: 0.42 for m in COMMON}
_AR_FLOOR.update({"max_sensitivity": 0.27, "random_logit": 0.27})


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=_GRID, lw=0.8)
    ax.set_axisbelow(True)


def make_reliability_bars() -> None:
    rel = pd.read_csv(f"{RES}/meta_evaluation_reliability.csv")
    rel = rel[rel["metric"].isin(COMMON)]
    g = (rel.groupby("metric")
            .agg(nr=("nr_score", "mean"), nr_sd=("nr_score", "std"),
                 ar=("ar_score", "mean"), ar_sd=("ar_score", "std"),
                 r=("combined_reliability", "mean"))
            .sort_values("r", ascending=False))
    x = np.arange(len(g))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(x - w / 2, g["nr"], w * 0.92, color=_BLUE, zorder=3,
           label="Noise Resilience (NR)")
    ax.bar(x + w / 2, g["ar"], w * 0.92, color=_ORANGE, hatch="////",
           edgecolor="white", lw=0, zorder=3, label="Adversarial Reactivity (AR)")
    for i, m in enumerate(g.index):
        f = _AR_FLOOR[m]
        ax.plot([i + w / 2 - w * 0.6, i + w / 2 + w * 0.6], [f, f],
                color=_INK, lw=1.6, zorder=5,
                label="AR noise floor (random ordering)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_SHORT[m] for m in g.index])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("score (mean over 14 model x method cells)", fontsize=8, color=_INK2)
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=3,
              bbox_to_anchor=(0.0, 1.0), handlelength=1.6, columnspacing=1.4)
    idx = list(g.index)
    for m, txt, dy in (("random_logit", "AR at its floor", 22),
                       ("relevance_mass_accuracy", "AR below its floor", 22)):
        i = idx.index(m)
        ax.annotate(txt, xy=(i + w / 2, g.loc[m, "ar"]),
                    xytext=(0, dy), textcoords="offset points", ha="center",
                    fontsize=7.5, color=_INK2,
                    arrowprops=dict(arrowstyle="-", color=_INK2, lw=0.6))
    plt.tight_layout()
    plt.savefig(f"{FIG}/reliability_bars.pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG}/reliability_bars.pdf")


def make_redundancy_bars(threshold: float = 0.85) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, model, lab in [(axes[0], "resnet18", "ResNet-18"),
                           (axes[1], "squeezenet", "SqueezeNet")]:
        r = pd.read_csv(f"{RES}/redundancy_matrix_{model}.csv", index_col=0)
        a = np.abs(r.values)
        iu = np.triu_indices_from(a, k=1)
        pairs = sorted([(a[i, j], r.index[i], r.columns[j]) for i, j in zip(*iu)],
                       reverse=True)
        vals = np.array([p[0] for p in pairs])
        n = len(vals)
        x = np.arange(n)
        pruned = vals > threshold
        ax.bar(x[~pruned], vals[~pruned], 0.8, color=_BLUE, zorder=3,
               label="retained pair")
        ax.bar(x[pruned], vals[pruned], 0.8, color=_ORANGE, hatch="////",
               edgecolor="white", lw=0, zorder=3, label="pruned pair")
        ax.axhline(threshold, color=_INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
        ax.text(n - 0.5, threshold + 0.015, f"pruning threshold |ρ| > {threshold:.2f}",
                ha="right", va="bottom", fontsize=7.5, color=_INK2)
        # Call out the two pruned pairs and the two nearest survivors with
        # leader lines into the empty upper-right area.
        ys = (1.06, 0.97, 0.76, 0.66)  # pruned above the line, survivors below
        for k in range(4):
            v, m1, m2 = pairs[k]
            ax.annotate(f"{_SHORT_ALL[m1]}–{_SHORT_ALL[m2]}  {v:.2f}",
                        xy=(k, v), xytext=(7.5, ys[k]),
                        textcoords="data", ha="left", va="center", fontsize=7.5,
                        color=_INK2,
                        arrowprops=dict(arrowstyle="-", color=_INK2, lw=0.6,
                                        shrinkA=0, shrinkB=2))
        ax.set_xticks([])
        ax.set_xlabel(f"{n} metric pairs, sorted by |ρ|", fontsize=8, color=_INK2)
        ax.set_title(lab, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        _style(ax)
        ax.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("Spearman |ρ| over (image, method) pairs", fontsize=8, color=_INK2)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right",
                   bbox_to_anchor=(1.0, 0.72), handlelength=1.6)
    plt.tight_layout()
    plt.savefig(f"{FIG}/redundancy_bars.pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG}/redundancy_bars.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    report_scalars()
    make_cd_diagram()
    make_ensemble_radar()
    make_ar_schematic()
    make_reliability_bars()
    make_redundancy_bars()
