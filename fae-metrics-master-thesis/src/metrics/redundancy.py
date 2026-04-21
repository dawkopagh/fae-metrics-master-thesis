"""
Metric redundancy analysis — Research Contribution 1.

Responsibilities (see docs/thesis_plan.md §2, Research Contribution 1;
and §10, Decision D4):
    - Compute a Spearman correlation matrix across all metrics for a given
      model × dataset × FAE-method combination
    - Apply the |ρ| > 0.85 (default, Decision D4) pruning threshold within
      each Quantus category to identify a non-redundant representative subset M*
    - Return the reduced metric set and the full correlation matrix for
      reporting in Chapter 4

Not responsible for: computing the metric scores themselves (see quantus_wrapper.py),
meta-validation weighting (see meta_evaluation/),
or score aggregation (see aggregation/).
"""
