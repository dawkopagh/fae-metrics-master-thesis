"""
Statistical significance tests for FAE method comparison — Research Contributions 3 & 4.

Responsibilities (see docs/thesis_plan.md §2, Research Contributions 3–4;
§6, Statistical Comparison; Decision D5):
    - Friedman test + Nemenyi post-hoc for ranking FAE methods across metrics
      on the full test set (non-parametric, multi-method comparison)
    - Wilcoxon signed-rank test for paired comparison of individual vs.
      NormEnsembleXAI-ensembled attributions per metric
    - Return corrected p-values, critical difference diagrams, and rank tables
      formatted for Chapter 4 figures

Not responsible for: computing effectiveness indices (see aggregation/),
metric score computation (see metrics/),
or visualisation beyond raw figure export (thesis LaTeX figures are authored
separately in Latex/).
"""
