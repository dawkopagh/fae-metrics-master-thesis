"""
Custom metric implementations not covered by or differing from quantus defaults.

Responsibilities (see docs/thesis_plan.md §6, Metric Computation):
    - Implement the Pointing Game metric using ISIC segmentation masks
      as a Localization baseline (reference: evaluate_xai_methods.ipynb)
    - Implement any metric variant whose quantus default hyperparameters
      diverge from the values chosen in Decision D7

Not responsible for: quantus-standard metrics (see quantus_wrapper.py),
redundancy analysis (see redundancy.py),
or aggregation (see aggregation/).
"""
