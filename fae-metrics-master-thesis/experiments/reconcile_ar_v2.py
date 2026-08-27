"""reconcile_ar_v2.py — post-Colab AR-v2 reconciliation harness.

Run AFTER copying the v2 artifacts from Drive into results/:
    meta_evaluation_reliability.csv   (patched by colab_ar_completion_v2.ipynb)
    ar_completion_v2_n64.csv          (per-cell AR + signed monotonicity)

It re-runs the local aggregation chain (compare_rankings ->
run_statistical_analysis) and prints a BEFORE/AFTER report of every
MQ-dependent number the thesis prose cites, so the LaTeX edit pass is
mechanical:

    - per-metric NR / AR / combined r_k (+ SD)      -> tab:meta, ch4/ch5/ch6
    - rho(AW, MQ) and rho(uniform, MQ) per model    -> tab:rank-stability,
                                                       ch4/ch5/ch6 prose
    - MQ-discounted per-method ranking per model    -> tab:ranking-*, LRP
                                                       position sentences
    - mean |Delta| MQ vs AW                          -> ch4/ch5/ch6 prose
    - Friedman chi^2 / Kendall's W (mqdiscount)      -> tab:friedman, ch6
    - direction audit of the v2 cells                -> ch4 audit sentence
                                                        (currently 4/42 v1)

Baseline numbers are snapshotted from the CURRENT results/ CSVs the first
time the script runs (--snapshot), so run it once BEFORE copying the v2
files, then again after. With identical inputs the report shows no changes
(self-test).

Usage:
    .venv/bin/python experiments/reconcile_ar_v2.py --snapshot   # before
    # ... copy v2 CSVs from Drive into results/ ...
    .venv/bin/python experiments/reconcile_ar_v2.py              # after
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent
_RESULTS = _REPO.parent / "results"
_SNAPSHOT = _RESULTS / "reconcile_ar_v2_baseline.json"
_PY = sys.executable

M_STAR = ["faithfulness_correlation", "max_sensitivity", "pixel_flipping",
          "pointing_game", "random_logit", "relevance_mass_accuracy",
          "sparseness"]


def _chain() -> None:
    """Re-run the local aggregation chain (rankings + stats)."""
    exp = _REPO / "experiments"
    for cmd in (
        [_PY, str(exp / "compare_rankings.py"),
         "--slice-csv", str(_RESULTS / "full_run_7fae_12metrics_600.csv"),
         "--reliability-csv", str(_RESULTS / "meta_evaluation_reliability.csv"),
         "--output-csv", str(_RESULTS / "ranking_comparison.csv")],
        [_PY, str(exp / "run_statistical_analysis.py"),
         "--ranking-csv", str(_RESULTS / "ranking_comparison.csv"),
         "--tests-out", str(_RESULTS / "statistical_tests.csv"),
         "--cd-out", str(_RESULTS / "cd_summary.csv")],
    ):
        print(f"$ {' '.join(Path(c).name if Path(c).exists() else c for c in cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def _state() -> dict:
    """Extract every prose-cited MQ-dependent number from results/."""
    s: dict = {}

    rel = pd.read_csv(_RESULTS / "meta_evaluation_reliability.csv")
    rel = rel[rel.metric.isin(M_STAR)]
    g = rel.groupby("metric")[["nr_score", "ar_score", "combined_reliability"]]
    means = g.mean().round(2)
    sds = rel.groupby("metric").combined_reliability.std().round(2)
    s["reliability"] = {
        m: {"NR": means.at[m, "nr_score"], "AR": means.at[m, "ar_score"],
            "r_k": means.at[m, "combined_reliability"], "SD": float(sds[m])}
        for m in means.index
    }

    rc = pd.read_csv(_RESULTS / "ranking_comparison.csv")
    mm = rc.groupby(["model", "fae_method"])[
        ["effectiveness_uniform", "effectiveness_autoweighted",
         "effectiveness_mqdiscount"]].mean()
    for model in ["resnet18", "squeezenet"]:
        sub = mm.loc[model]
        s[f"rho_aw_mq_{model}"] = round(float(spearmanr(
            sub.effectiveness_autoweighted, sub.effectiveness_mqdiscount
        ).statistic), 3)
        s[f"rho_uni_mq_{model}"] = round(float(spearmanr(
            sub.effectiveness_uniform, sub.effectiveness_mqdiscount
        ).statistic), 3)
        order = sub.effectiveness_mqdiscount.sort_values(ascending=False)
        s[f"mq_ranking_{model}"] = list(order.index)
        s[f"mq_values_{model}"] = [round(v, 3) for v in order.values]
        s[f"lrp_mq_pos_{model}"] = s[f"mq_ranking_{model}"].index("lrp") + 1
        s[f"mean_abs_delta_{model}"] = round(float(
            (sub.effectiveness_mqdiscount
             - sub.effectiveness_autoweighted).abs().mean()), 3)

    st = pd.read_csv(_RESULTS / "statistical_tests.csv")
    fr = st[(st.test == "friedman") & (st.scheme == "mqdiscount")]
    for r in fr.itertuples():
        s[f"friedman_mq_{r.model}"] = {"chi2": round(r.statistic, 1),
                                       "kendalls_w": round(r.effect_size, 3)}

    v2p = _RESULTS / "ar_completion_v2_n64.csv"
    if v2p.exists():
        v2 = pd.read_csv(v2p)
        s["v2_direction_audit"] = {
            "n_cells": int(len(v2)),
            "n_wrong_direction": int(v2.get(
                "wrong_direction", pd.Series(dtype=bool)).sum()),
            "mean_AR": {m: round(float(g.ar_score.mean()), 3)
                        for m, g in v2.groupby("metric")},
        }
    return s


def _is_nan(v) -> bool:
    try:
        return isinstance(v, float) and np.isnan(v)
    except TypeError:
        return False


def _diff(before: dict, after: dict, prefix: str = "") -> list[str]:
    lines = []
    keys = sorted(set(before) | set(after), key=str)
    for k in keys:
        b, a = before.get(k), after.get(k)
        if isinstance(b, dict) and isinstance(a, dict):
            lines += _diff(b, a, prefix=f"{prefix}{k}.")
        elif _is_nan(b) and _is_nan(a):
            continue  # NaN == NaN for diff purposes (JSON round-trips NaN)
        elif b != a:
            lines.append(f"  {prefix}{k}: {b}  ->  {a}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot", action="store_true",
                   help="Save the current numbers as the BEFORE baseline "
                        "and exit (run once before copying the v2 CSVs).")
    p.add_argument("--no-chain", action="store_true",
                   help="Skip re-running the aggregation chain (report only).")
    args = p.parse_args()

    if args.snapshot:
        _SNAPSHOT.write_text(json.dumps(_state(), indent=1, default=str))
        print(f"Baseline snapshot written -> {_SNAPSHOT}")
        return

    if not _SNAPSHOT.exists():
        sys.exit("No baseline snapshot. Run with --snapshot first "
                 "(BEFORE copying the v2 CSVs into results/).")

    if not args.no_chain:
        _chain()

    before = json.loads(_SNAPSHOT.read_text())
    after = _state()
    changes = _diff(before, after)

    print("\n===== AR-v2 reconciliation report =====")
    if not changes:
        print("No MQ-dependent numbers changed (inputs identical to baseline).")
    else:
        print(f"{len(changes)} changed values:")
        print("\n".join(changes))
        print("""
PROSE SPOTS TO RECONCILE (grep the old value; regenerate tables first):
  tables:   render_results_tables.py --n-images-label "full run, 600 images"
            (tab:meta caption: update the 12-image extended-AR note to the
             v2 protocol: same 64-image sample, 10 levels)
  figures:  make_figures.py (CD diagram if mqdiscount ranks moved)
  ch4: sec:exp-meta reliability narrative + jackknife/audit sentences
       (4/42 -> v2 audit), extending-AR paragraph (rho, LRP position),
       sec:exp-ranking MQ column sentences, intro item (i) mean |Delta|
  ch5: MQ-discount bullet (rho, LRP, r_k values), extended-AR threat
       (sample-mixing sentence NO LONGER applies if v2 used the 64 sample)
  ch6: Contribution 2 verdict (rho, LRP, |Delta|), Contribution 4 chi^2/W
  abstract + Streszczenie: reliability ordering sentence if r_k order moved
  memory: pixel-flipping-direction-bug.md + latex-outstanding-issues.md
Then: rebuild praca.pdf twice and re-run this script (report must be empty).""")


if __name__ == "__main__":
    main()
