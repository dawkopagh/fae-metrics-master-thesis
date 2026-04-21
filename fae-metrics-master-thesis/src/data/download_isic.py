"""
Download and organise the ISIC 2017 Task 1 dataset from challenge.isic-archive.com.

Downloads image ZIPs, segmentation mask ZIPs, and diagnosis ground-truth CSVs
from the public S3 bucket, then reorganises into the class-folder layout
expected by ``torchvision.datasets.ImageFolder`` and ``ISIC2017Dataset``.

Target layout::

    {dest_dir}/
      images/{train,validation,test}/{melanoma,nevus,seborrheic_keratosis}/
      masks/{train,validation,test}/{melanoma,nevus,seborrheic_keratosis}/
"""

from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_S3_BASE = "https://isic-challenge-data.s3.amazonaws.com/2017"

# (split_name, images_zip, masks_zip, diagnosis_csv)
_SPLIT_CONFIG = [
    (
        "train",
        f"{_S3_BASE}/ISIC-2017_Training_Data.zip",
        f"{_S3_BASE}/ISIC-2017_Training_Part1_GroundTruth.zip",
        f"{_S3_BASE}/ISIC-2017_Training_Part3_GroundTruth.csv",
    ),
    (
        "validation",
        f"{_S3_BASE}/ISIC-2017_Validation_Data.zip",
        f"{_S3_BASE}/ISIC-2017_Validation_Part1_GroundTruth.zip",
        f"{_S3_BASE}/ISIC-2017_Validation_Part3_GroundTruth.csv",
    ),
    (
        "test",
        f"{_S3_BASE}/ISIC-2017_Test_v2_Data.zip",
        f"{_S3_BASE}/ISIC-2017_Test_v2_Part1_GroundTruth.zip",
        f"{_S3_BASE}/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
    ),
]

_EXPECTED_COUNTS = {"train": 2000, "validation": 150, "test": 600}

CLASS_NAMES = ["melanoma", "nevus", "seborrheic_keratosis"]


def _download_file(url: str, dest: Path, skip_existing: bool = True) -> Path:
    """Download a file from *url* to *dest* with a progress bar.

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Local destination path.
    skip_existing : bool
        If True and *dest* exists, skip the download.

    Returns
    -------
    Path
        The destination path.
    """
    if skip_existing and dest.exists():
        logger.info("Skipping (exists): %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url.split("/")[-1], dest)

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def _parse_diagnosis_csv(csv_url: str) -> dict[str, str]:
    """Download and parse a Part-3 ground-truth CSV.

    Parameters
    ----------
    csv_url : str
        URL of the CSV with columns ``image_id, melanoma, seborrheic_keratosis``.

    Returns
    -------
    dict[str, str]
        Mapping from image_id to class name.
    """
    resp = requests.get(csv_url, timeout=60)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    labels: dict[str, str] = {}
    for row in reader:
        image_id = row["image_id"].strip()
        if float(row["melanoma"]) == 1.0:
            labels[image_id] = "melanoma"
        elif float(row["seborrheic_keratosis"]) == 1.0:
            labels[image_id] = "seborrheic_keratosis"
        else:
            labels[image_id] = "nevus"
    return labels


def _extract_and_organise(
    zip_path: Path,
    output_dir: Path,
    labels: dict[str, str],
    is_mask: bool = False,
) -> int:
    """Extract a ZIP and move files into class sub-directories.

    Parameters
    ----------
    zip_path : Path
        Path to the downloaded ZIP.
    output_dir : Path
        Target directory (e.g. ``dest/images/train/``).
    labels : dict[str, str]
        Image-ID → class-name mapping.
    is_mask : bool
        If True, derive the image_id from mask filenames
        (``ISIC_XXXXXXX_segmentation.png``).

    Returns
    -------
    int
        Number of files extracted and placed.
    """
    for cn in CLASS_NAMES:
        (output_dir / cn).mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            # Skip directories, macOS resource forks, and non-image files
            if not basename or basename.startswith("._") or basename.startswith("__"):
                continue
            if not basename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Derive image_id
            if is_mask:
                # ISIC_XXXXXXX_segmentation.png → ISIC_XXXXXXX
                image_id = basename.replace("_segmentation", "").rsplit(".", 1)[0]
            else:
                image_id = basename.rsplit(".", 1)[0]

            class_name = labels.get(image_id)
            if class_name is None:
                logger.warning("No label for %s — skipping", image_id)
                continue

            dest_file = output_dir / class_name / basename
            if dest_file.exists():
                count += 1
                continue

            with zf.open(member) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1

    return count


def download_isic2017(
    dest_dir: str = "data/isic2017",
    skip_existing: bool = True,
) -> None:
    """Download and organise the full ISIC 2017 dataset.

    Parameters
    ----------
    dest_dir : str
        Root output directory. Will contain ``images/`` and ``masks/``
        sub-trees organised by split and class.
    skip_existing : bool
        If True, skip downloads and extractions that have already completed.
    """
    dest = Path(dest_dir)
    zip_dir = dest / "_downloads"
    zip_dir.mkdir(parents=True, exist_ok=True)

    for split_name, images_url, masks_url, csv_url in _SPLIT_CONFIG:
        logger.info("=== Processing split: %s ===", split_name)

        # 1. Parse diagnosis labels
        logger.info("Fetching ground-truth labels...")
        labels = _parse_diagnosis_csv(csv_url)
        logger.info("  %d labels loaded", len(labels))

        # 2. Download and organise images
        images_zip = _download_file(
            images_url, zip_dir / f"{split_name}_images.zip", skip_existing
        )
        images_dir = dest / "images" / split_name
        n_images = _extract_and_organise(images_zip, images_dir, labels, is_mask=False)
        logger.info("  Images: %d files in %s", n_images, images_dir)

        # 3. Download and organise masks
        masks_zip = _download_file(
            masks_url, zip_dir / f"{split_name}_masks.zip", skip_existing
        )
        masks_dir = dest / "masks" / split_name
        n_masks = _extract_and_organise(masks_zip, masks_dir, labels, is_mask=True)
        logger.info("  Masks:  %d files in %s", n_masks, masks_dir)

        # 4. Verify counts
        expected = _EXPECTED_COUNTS[split_name]
        if n_images != expected:
            logger.warning(
                "  Expected %d images for %s, got %d", expected, split_name, n_images
            )

    logger.info("ISIC 2017 download complete → %s", dest)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    download_isic2017()
