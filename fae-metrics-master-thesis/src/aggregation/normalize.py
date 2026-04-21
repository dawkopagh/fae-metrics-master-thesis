"""
Per-metric normalisation of raw quantus scores before aggregation.

Responsibilities (see docs/thesis_plan.md §6, Aggregation; Decision D1):
    - Implement Second Moment Scaling (default, Hryniewska-Guzik et al., 2024):
      s̃ = s / sqrt(E[s²])
    - Implement z-score and min-max normalisation for ablation comparisons
    - Apply direction rectification so that higher score always means better
      for all metrics (some quantus metrics are lower-is-better by convention)
    - Accept a long-format DataFrame of raw scores; return the same schema
      with a normalised_score column appended

Not responsible for: computing raw scores (see metrics/), weighting
(see weighting.py), or the final aggregation operator (see aggregate.py).
"""
