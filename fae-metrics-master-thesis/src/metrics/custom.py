"""
Custom localization metrics — INTENTIONALLY OUT OF SCOPE (documented placeholder).

Original intent (docs/thesis_plan.md §6, Metric Computation):
    - Implement the Pointing Game metric using ISIC segmentation masks
      as a Localization baseline (reference: evaluate_xai_methods.ipynb)
    - Implement any metric variant whose quantus default hyperparameters
      diverge from the values chosen in Decision D7

Decision (this module is deliberately empty)
---------------------------------------------
After wiring the metric layer in ``src/metrics/quantus_wrapper.py``, the two
Localization metrics in the 12-metric set are already provided by Quantus and
configured to consume the ISIC binary segmentation masks:

    * ``quantus.PointingGame``          — fed ``s_batch`` (the ISIC mask) and a
                                          single-channel attribution, abs=True.
    * ``quantus.RelevanceMassAccuracy`` — fed the same ``s_batch`` and a
                                          single-channel attribution.

(See ``compute_all_metrics`` in quantus_wrapper.py, the "LOCALIZATION"
section.) Quantus' implementations match the canonical definitions used in the
reference notebook, accept the ISIC masks directly, and respect the direction
conventions registered in ``src/aggregation/normalize.METRIC_DIRECTIONS``
(both Localization metrics are +1, higher-is-better). The Decision-D7
hyperparameters for these metrics are the Quantus defaults, so no metric
variant / override is required either.

Re-implementing Pointing Game (or a sign/direction override) here would
therefore duplicate Quantus with no methodological benefit and would add a
second code path to validate against the same masks — strictly higher risk.
This module is consequently kept as a documented, intentionally empty
placeholder rather than a bare stub, so the thesis methodology (Chapter 3) can
state explicitly that **all localization metrics are sourced from Quantus and
no custom localization metric was needed**.

When to revisit
---------------
Add an implementation here only if a future Localization metric is required
that Quantus genuinely lacks, or if a sign/direction override deviating from
Quantus' convention becomes necessary. In that case, mirror the type hints,
docstrings, and logging style of ``redundancy.py`` / ``weighting.py`` and add
matching tests under ``tests/``.

Not responsible for: quantus-standard metrics (see quantus_wrapper.py),
redundancy analysis (see redundancy.py),
or aggregation (see aggregation/).
"""

from __future__ import annotations

# Intentionally no implementation — see module docstring.
# All localization metrics are provided by src/metrics/quantus_wrapper.py.
__all__: list[str] = []
