"""
Download and organise the ISIC 2017 Task 1 dataset from challenge.isic-archive.com.

Responsibilities (see docs/thesis_plan.md §3, Datasets):
    - Download the 2,000-train / 150-val / 600-test image archives and
      the corresponding segmentation mask archives via the ISIC API
    - Unpack archives into data/isic2017/{images,masks}/{train,val,test}/{class}/
    - Rename files to the canonical ISIC_XXXXXXX[_segmentation].{jpg,png} convention
    - Log download checksums for reproducibility

Not responsible for: Dataset class construction (see isic_dataset.py) or
any model training (see models/train.py).
"""
