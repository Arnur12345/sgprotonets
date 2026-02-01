"""One-time data preprocessing script for IU Chest X-Ray dataset.

Downloads (if needed) and converts the IU Chest X-Ray dataset into a
standardized format: metadata.csv + images/ directory.

Expected raw structure:
    data_dir/
        images/
            CXR1_1_IM-0001-3001.png
            ...
        indiana_reports.csv  (or reports XML)
        indiana_projections.csv

Output structure:
    output_dir/
        images/  (symlinked or copied)
        metadata.csv  (columns: image_path, report, label)
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def extract_label(findings: str, impression: str) -> str:
    """Extract dominant pathology label from report text.

    Args:
        findings: Findings section text.
        impression: Impression section text.

    Returns:
        Single label string (dominant finding).
    """
    text = f"{findings} {impression}".lower()

    # Priority-ordered label detection
    label_keywords = {
        "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
        "effusion": ["effusion", "pleural fluid"],
        "infiltrate": ["infiltrat"],
        "mass": ["mass", "lesion"],
        "nodule": ["nodule"],
        "pneumonia": ["pneumonia"],
        "atelectasis": ["atelectasis", "atelectatic"],
        "consolidation": ["consolidation", "consolidative"],
        "pneumothorax": ["pneumothorax"],
        "edema": ["edema", "oedema"],
        "emphysema": ["emphysema", "hyperinflat"],
        "hernia": ["hernia"],
        "fibrosis": ["fibrosis", "fibrotic"],
        "pleural_thickening": ["pleural thickening", "pleural thick"],
    }

    for label, keywords in label_keywords.items():
        for kw in keywords:
            if kw in text:
                return label

    return "no_finding"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess IU Chest X-Ray dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw IU X-Ray data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reports
    reports_path = data_dir / "indiana_reports.csv"
    if not reports_path.exists():
        logger.error(f"Reports file not found: {reports_path}")
        sys.exit(1)

    reports_df = pd.read_csv(reports_path)
    logger.info(f"Loaded {len(reports_df)} reports")

    # Load projections (image metadata)
    projections_path = data_dir / "indiana_projections.csv"
    if projections_path.exists():
        proj_df = pd.read_csv(projections_path)
        # Filter to frontal views only
        proj_df = proj_df[proj_df["projection"].str.contains("frontal", case=False, na=False)]
    else:
        proj_df = None
        logger.warning("No projections file found, using all images")

    # Build metadata
    records = []
    images_dir = data_dir / "images"

    for _, row in reports_df.iterrows():
        uid = row.get("uid", row.get("study_id", ""))
        findings = str(row.get("findings", ""))
        impression = str(row.get("impression", ""))
        report = f"{findings} {impression}".strip()

        # Find matching image(s)
        if proj_df is not None:
            matching = proj_df[proj_df["uid"] == uid]
            image_files = matching["filename"].tolist()
        else:
            # Try to find by UID pattern
            image_files = [f.name for f in images_dir.glob(f"*{uid}*")]

        label = extract_label(findings, impression)

        for img_file in image_files:
            img_path = f"images/{img_file}"
            if (data_dir / img_path).exists() or (output_dir / img_path).exists():
                records.append({
                    "image_path": img_path,
                    "report": report,
                    "label": label,
                })

    metadata = pd.DataFrame(records)
    logger.info(f"Created metadata with {len(metadata)} samples")
    logger.info(f"Label distribution:\n{metadata['label'].value_counts()}")

    # Save
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    # Symlink images directory
    output_images = output_dir / "images"
    if not output_images.exists():
        output_images.symlink_to(images_dir.resolve())
        logger.info(f"Symlinked images: {output_images} -> {images_dir}")

    logger.info(f"Preprocessing complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
