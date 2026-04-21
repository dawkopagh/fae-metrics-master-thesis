"""
Metric selection: combines redundancy pruning and meta-validation to produce M*.

Responsibilities (see docs/thesis_plan.md §6, Meta-Evaluation):
    - Accept the correlation matrix from redundancy.py and the meta-validation
      scores from metaquantus_wrapper.py
    - Apply the two-stage filter: (1) discard redundant metrics per category,
      (2) flag metrics with low NR or AR scores for reporting
    - Return the final validated metric set M* used by the aggregation layer

Not responsible for: computing correlation or meta-validation scores directly
(those are delegated to redundancy.py and metaquantus_wrapper.py),
or the aggregation of FAE method scores (see aggregation/).
"""
