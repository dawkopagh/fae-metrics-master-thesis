"""run_statistical_analysis.py — Statistical significance tests for Chapter 4.

LOCAL post-processing script (runs on the author's machine via the venv, NOT
Colab). Ingests the ranking-comparison CSV (and, when present, the ensemble
ranking CSV) produced by the full Colab run and runs:

  * Friedman omnibus test + Nemenyi post-hoc across the 7 FAE methods, per
    model and per weighting scheme (uniform / autoweighted / mqdiscount /
    single_fc). Drives Table~\\ref{tab:friedman} and the CD diagram
    (Figure~\\ref{fig:cd-diagram}) of chapter4_state_observers.tex.
  * Wilcoxon signed-rank test for the paired individual-vs-ensembled
    comparison, per model. Drives Table~\\ref{tab:wilcoxon}.

It uses the new src/comparison/statistical_tests.py helpers exclusively.

Outputs
-------
results/statistical_tests.csv
    Long-format table of every test run. One row per test with columns:
    test, model, scheme, comparison, statistic, p_value, n, effect_size,
    significant_0.05, extra.
results/cd_summary.csv
    Per-method mean-rank / critical-difference summary (one block per
    model x scheme), built from build_cd_summary(). Feeds the CD diagram.

The script runs cleanly on the 12-image PILOT CSVs in results/ (the pilot has
7 FAE methods, satisfying Friedman's k>=3 requirement) and equally on the
future 600-image FULL CSVs — just point --ranking-csv / --ensemble-csv at the
*_600 / *_FULL files.

Exact command (pilot, defaults point at results/):

    fae-metrics-master-thesis/.venv/bin/python \\
        fae-metrics-master-thesis/experiments/run_statistical_analysis.py

Full run:

    .venv/bin/python experiments/run_statistical_analysis.py \\
        --ranking-csv  ../results/ranking_comparison_FULL.csv \\
        --ensemble-csv ../results/ensemble_ranking_FULL.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.comparison.statistical_tests import (  # noqa: E402
    build_cd_summary,
    friedman_test,
    nemenyi_posthoc,
    wilcoxon_paired,
)

# Repository-root results dir (one level above the code subrepo).
_RESULTS = _REPO.parent / "results"

# Effectiveness columns in the ranking-comparison CSV → human scheme label.
_SCHEMES: dict[str, str] = {
    "effectiveness_uniform": "uniform",
    "effectiveness_autoweighted": "autoweighted",
    "effectiveness_mqdiscount": "mqdiscount",
    "effectiveness_single_fc": "single_fc",
}

_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Friedman + Nemenyi across FAE methods
# ---------------------------------------------------------------------------

def _run_friedman_nemenyi(
    ranking_df: pd.DataFrame,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Friedman + Nemenyi per (model, scheme). Returns (test_rows, cd_frames)."""
    test_rows: list[dict] = []
    cd_frames: list[pd.DataFrame] = []

    models = sorted(ranking_df["model"].unique())
    for model in models:
        sub = ranking_df[ranking_df["model"] == model]
        for col, scheme in _SCHEMES.items():
            if col not in sub.columns:
                continue
            # Blocks are images (one model fixed) → block on image_id only.
            try:
                fres = friedman_test(
                    sub,
                    method_col="fae_method",
                    block_cols=("image_id",),
                    value_col=col,
                    higher_is_better=True,
                )
            except ValueError as exc:
                print(f"  [skip] Friedman {model}/{scheme}: {exc}")
                continue

            sig = bool(fres["p_value"] < _ALPHA) if np.isfinite(fres["p_value"]) else False
            test_rows.append({
                "test": "friedman",
                "model": model,
                "scheme": scheme,
                "comparison": "fae_methods",
                "statistic": fres["statistic"],
                "p_value": fres["p_value"],
                "n": fres["n_blocks"],
                "effect_size": np.nan,
                "significant_0.05": sig,
                "extra": f"k={fres['n_methods']};dropped={fres['n_blocks_dropped']}",
            })

            # Nemenyi post-hoc (always compute CD/ranks; only meaningful when
            # the omnibus is significant, but the table is harmless otherwise).
            try:
                nres = nemenyi_posthoc(
                    sub,
                    method_col="fae_method",
                    block_cols=("image_id",),
                    value_col=col,
                    alpha=_ALPHA,
                    higher_is_better=True,
                )
            except ValueError as exc:
                print(f"  [skip] Nemenyi {model}/{scheme}: {exc}")
                nres = None

            cd = build_cd_summary(fres, nres)
            cd.insert(0, "model", model)
            cd.insert(1, "scheme", scheme)
            cd_frames.append(cd)

    return test_rows, cd_frames


# ---------------------------------------------------------------------------
# Wilcoxon: individual vs ensembled (paired per image)
# ---------------------------------------------------------------------------

def _individual_best_per_image(
    ranking_df: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """Per (model, image_id) best individual effectiveness under *value_col*."""
    grp = (
        ranking_df.groupby(["model", "image_id"])[value_col]
        .max()
        .reset_index()
        .rename(columns={value_col: "individual_best"})
    )
    return grp


def _run_wilcoxon_ensemble(
    ranking_df: pd.DataFrame,
    ensemble_df: pd.DataFrame | None,
    value_col: str = "effectiveness_mqdiscount",
) -> list[dict]:
    """Wilcoxon ensemble-vs-best-individual, paired per image, per model.

    The ensemble CSV is expected in the same wide schema as the ranking CSV
    (model, fae_method='ensemble', image_id, effectiveness_*). When it is
    absent (pilot has no ensemble run yet) the test is skipped with a note.
    """
    if ensemble_df is None or ensemble_df.empty:
        print("  [skip] Wilcoxon ensemble vs individual: no ensemble CSV "
              "(expected after the full run).")
        return []

    rows: list[dict] = []
    best_ind = _individual_best_per_image(ranking_df, value_col)

    for model in sorted(ensemble_df["model"].unique()):
        ens = ensemble_df[ensemble_df["model"] == model][
            ["image_id", value_col]
        ].rename(columns={value_col: "ensemble"})
        ind = best_ind[best_ind["model"] == model]
        merged = ind.merge(ens, on="image_id", how="inner")
        if merged.empty:
            print(f"  [skip] Wilcoxon {model}: no matched images.")
            continue
        try:
            wres = wilcoxon_paired(
                merged["ensemble"].to_numpy(),
                merged["individual_best"].to_numpy(),
                alternative="two-sided",
            )
        except ValueError as exc:
            print(f"  [skip] Wilcoxon {model}: {exc}")
            continue

        sig = bool(wres["p_value"] < _ALPHA) if np.isfinite(wres["p_value"]) else False
        rows.append({
            "test": "wilcoxon",
            "model": model,
            "scheme": value_col.replace("effectiveness_", ""),
            "comparison": "ensemble_vs_best_individual",
            "statistic": wres["statistic"],
            "p_value": wres["p_value"],
            "n": wres["n"],
            "effect_size": wres["effect_size"],
            "significant_0.05": sig,
            "extra": f"direction={wres['direction']};n_zero={wres['n_zero']}",
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Friedman/Nemenyi + Wilcoxon for Chapter 4 (local).",
    )
    p.add_argument(
        "--ranking-csv",
        default=str(_RESULTS / "ranking_comparison.csv"),
        metavar="PATH",
        help="Four-scheme ranking CSV (default: pilot results/ranking_comparison.csv).",
    )
    p.add_argument(
        "--ensemble-csv",
        default=str(_RESULTS / "ensemble_ranking.csv"),
        metavar="PATH",
        help=(
            "Ensemble ranking CSV (same wide schema, fae_method='ensemble'). "
            "Optional — Wilcoxon is skipped if missing "
            "(default: results/ensemble_ranking.csv)."
        ),
    )
    p.add_argument(
        "--wilcoxon-scheme",
        default="effectiveness_mqdiscount",
        metavar="COL",
        help="Effectiveness column used for the ensemble Wilcoxon test.",
    )
    p.add_argument(
        "--tests-out",
        default=str(_RESULTS / "statistical_tests.csv"),
        metavar="PATH",
        help="Output CSV for all test rows (default: results/statistical_tests.csv).",
    )
    p.add_argument(
        "--cd-out",
        default=str(_RESULTS / "cd_summary.csv"),
        metavar="PATH",
        help="Output CSV for the rank/CD summary (default: results/cd_summary.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ranking_path = Path(args.ranking_csv)
    ensemble_path = Path(args.ensemble_csv)

    if not ranking_path.exists():
        sys.exit(f"Ranking CSV not found: {ranking_path}")

    ranking_df = pd.read_csv(ranking_path)
    print(f"Loaded ranking CSV: {len(ranking_df)} rows from {ranking_path}")

    ensemble_df: pd.DataFrame | None = None
    if ensemble_path.exists():
        ensemble_df = pd.read_csv(ensemble_path)
        print(f"Loaded ensemble CSV: {len(ensemble_df)} rows from {ensemble_path}")
    else:
        print(f"Ensemble CSV not found ({ensemble_path}); Wilcoxon will be skipped.")

    print("\n=== Friedman + Nemenyi across FAE methods ===")
    friedman_rows, cd_frames = _run_friedman_nemenyi(ranking_df)

    print("\n=== Wilcoxon: ensemble vs best individual ===")
    wilcoxon_rows = _run_wilcoxon_ensemble(
        ranking_df, ensemble_df, value_col=args.wilcoxon_scheme
    )

    # --- Write tests CSV ---
    all_rows = friedman_rows + wilcoxon_rows
    tests_df = pd.DataFrame(all_rows, columns=[
        "test", "model", "scheme", "comparison", "statistic", "p_value",
        "n", "effect_size", "significant_0.05", "extra",
    ])
    Path(args.tests_out).parent.mkdir(parents=True, exist_ok=True)
    tests_df.to_csv(args.tests_out, index=False)
    print(f"\nWrote {len(tests_df)} test rows → {args.tests_out}")

    # --- Write CD summary CSV ---
    if cd_frames:
        cd_df = pd.concat(cd_frames, ignore_index=True)
    else:
        cd_df = pd.DataFrame(columns=[
            "model", "scheme", "method", "mean_rank", "rank_position",
            "critical_difference", "friedman_statistic", "friedman_p_value",
            "n_blocks", "n_methods",
        ])
    cd_df.to_csv(args.cd_out, index=False)
    print(f"Wrote {len(cd_df)} CD-summary rows → {args.cd_out}")

    # --- Console summary ---
    if not tests_df.empty:
        print("\n--- Test summary ---")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(tests_df.to_string(index=False))


if __name__ == "__main__":
    main()
