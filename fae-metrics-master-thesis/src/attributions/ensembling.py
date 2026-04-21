"""
NormEnsembleXAI wrapper: ensemble multiple attribution maps into a single map.

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 3;
and §10, Decision D6):
    - Implement the NormEnsembleXAI Mean aggregation with Second Moment
      Scaling normalisation (Hryniewska-Guzik et al., 2024) as the
      default ensembling strategy
    - Accept a dict of {fae_method: attribution_map} and return one
      ensembled map in the canonical (H, W) shape
    - Support alternative aggregation operators (sum, max) for ablation

Not responsible for: generating individual attributions (see generate.py),
metric computation on ensembled maps (see metrics/),
or the multi-criteria aggregation of metric scores (see aggregation/).
"""
