"""
Weight computation for the aggregation layer — Research Contribution 2 (novel part).

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 2;
§6, Aggregation; Decision D3):
    - Implement Autoweighted weighting (NormEnsembleXAI, Hryniewska-Guzik 2024):
      weights derived from inter-metric variance across FAE methods
    - Implement the MetaQuantus-discounted extension: multiply Autoweighted
      coefficients by the NR × AR meta-validation score for each metric
      (this is the novel contribution of the thesis)
    - Return a per-metric weight vector w ∈ ℝ^|M*|

Not responsible for: normalisation (see normalize.py),
aggregation itself (see aggregate.py),
or computing meta-validation scores (see meta_evaluation/).
"""
