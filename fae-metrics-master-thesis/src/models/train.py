"""
Training loop for the base classifiers on ISIC 2017.

Responsibilities (see docs/thesis_plan.md §4, Models Under Evaluation):
    - Train ResNet-18 and SqueezeNet 1.1 with class-weighted cross-entropy
    - Use fixed random seeds (torch, numpy, random) for reproducibility
    - Persist best-validation-accuracy checkpoint to weights/
    - Log per-epoch loss and accuracy to a results/ CSV

Not responsible for: model architecture definitions (see classifiers.py),
attribution generation (see attributions/generate.py),
or any evaluation metrics (see metrics/).
"""
