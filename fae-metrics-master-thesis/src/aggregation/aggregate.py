"""
Aggregation operators that combine normalised metric scores into a scalar
effectiveness index E(Φ) per FAE method per model.

Responsibilities (see docs/thesis_plan.md §6, Aggregation; Decision D2):
    - Implement weighted mean aggregation (default, recommended in thesis)
    - Implement min aggregation and geometric mean aggregation for comparison
    - Accept normalised scores and per-metric weights; return one E(Φ) scalar
      per (model, fae_method) pair
    - Produce a results DataFrame with all three operator scores for reporting

Not responsible for: normalisation (see normalize.py),
weight computation (see weighting.py),
or statistical comparison across FAE methods (see comparison/).
"""
