"""
Wrapper over MetaQuantus for meta-validation of individual evaluation metrics.

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 2;
and §6, Meta-Evaluation):
    - Run the MetaQuantus NR (Noise Resilience) and AR (Attribution Randomisation)
      tests (Hedström et al., 2024) for each metric in the candidate set
    - Return per-metric meta-validation scores (NR, AR) used as discount factors
      in the Autoweighted aggregation (Decision D3)
    - Interface with quantus_wrapper.py to avoid recomputing base metric scores

Not responsible for: selecting which metrics survive pruning (see metric_selection.py),
computing the base metric scores (see metrics/quantus_wrapper.py),
or the final aggregation (see aggregation/).
"""
