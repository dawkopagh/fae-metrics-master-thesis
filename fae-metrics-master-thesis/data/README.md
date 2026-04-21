# Data Directory

This directory is populated by `src/data/download_isic.py`, which
downloads the full ISIC 2017 Challenge Task 1 (Lesion Segmentation)
dataset from the ISIC archive. Contents are gitignored because the
full dataset is ~24 GB.

## Expected layout after download
data/
├── images/
│   ├── train/{melanoma,nevus,seborrheic_keratosis}/.jpg
│   ├── validation/{melanoma,nevus,seborrheic_keratosis}/.jpg
│   └── test/{melanoma,nevus,seborrheic_keratosis}/.jpg
└── masks/
├── train/{melanoma,nevus,seborrheic_keratosis}/_segmentation.png
├── validation/...
└── test/...

## To reproduce

```bash
python src/data/download_isic.py --target-dir data/
```

Counts: 2,000 train / 150 val / 600 test images, each with paired mask.
Total: ~2,750 images + 2,750 masks.

## Source

ISIC 2017 Challenge, Task 1 — Lesion Segmentation:
https://challenge.isic-archive.com/data/#2017

Citation: Codella et al. 2017 (arXiv:1710.05006).
License: CC-BY-NC.
