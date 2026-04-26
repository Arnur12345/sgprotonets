"""Build a metadata CSV for the BreaKHis_v1 dataset.

Crawls the standard BreaKHis_v1 folder tree, validates file names,
and writes a CSV that summarises every image with its subtype, binary
class, magnification, and patient identifier.  The script also prints
a concise statistics report to stdout so you can verify the dataset
before training.

The output CSV is optional — BreakHisDataset crawls the tree directly
at runtime — but it is useful for:

* Quick sanity checks and dataset statistics.
* Planning patient-level train / val splits (to prevent data leakage
  across patients).
* Cross-referencing with VLM-generated reports from
  ``generate_breakhis_reports.py``.

Output CSV columns
------------------
image_path      Relative to ``data_dir`` (forward slashes).
subtype         Canonical subtype name, e.g. ``ductal_carcinoma``.
binary_label    ``benign`` or ``malignant``.
magnification   ``40X``, ``100X``, ``200X``, or ``400X``.
patient_id      Slide identifier parsed from the filename, e.g. ``14-4659``.
filename        Basename only, e.g. ``SOB_B_TA-14-4659-40-001.png``.

Usage
-----
::

    python scripts/preprocess_breakhis.py \\
        --data_dir /path/to/BreaKHis_v1 \\
        --output_csv data/breakhis_metadata.csv

    # Stats only, no CSV written:
    python scripts/preprocess_breakhis.py \\
        --data_dir /path/to/BreaKHis_v1 \\
        --stats_only
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.breakhis import (
    BENIGN_SUBTYPES,
    MALIGNANT_SUBTYPES,
    VALID_MAGNIFICATIONS,
    parse_breakhis_filename,
)

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "image_path",
    "subtype",
    "binary_label",
    "magnification",
    "patient_id",
    "filename",
]


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

def crawl(data_dir: Path) -> tuple[list[dict], list[str]]:
    """Walk the BreaKHis_v1 tree and collect one record per image.

    Args:
        data_dir: Root of the BreaKHis_v1 download (contains
            ``histology_slides/``).

    Returns:
        ``(records, skipped)`` where ``records`` is a list of row dicts
        and ``skipped`` is a list of paths that failed to parse.
    """
    slides_root = data_dir / "histology_slides" / "breast"
    records: list[dict] = []
    skipped: list[str] = []

    for class_name in ("benign", "malignant"):
        sob_dir = slides_root / class_name / "SOB"
        if not sob_dir.is_dir():
            logger.warning(f"Expected directory not found: {sob_dir}")
            continue

        for subtype_dir in sorted(sob_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue

            for patient_dir in sorted(subtype_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue

                for mag_dir in sorted(patient_dir.iterdir()):
                    if not mag_dir.is_dir():
                        continue

                    for img_file in sorted(mag_dir.glob("*.png")):
                        parsed = parse_breakhis_filename(img_file.name)
                        if parsed is None:
                            skipped.append(str(img_file.relative_to(data_dir)))
                            logger.debug(
                                f"Skipping unrecognised filename: {img_file.name}"
                            )
                            continue

                        rel_path = str(img_file.relative_to(data_dir))
                        records.append({
                            "image_path":   rel_path,
                            "subtype":      parsed["subtype"],
                            "binary_label": class_name,
                            "magnification": parsed["magnification"],
                            "patient_id":   parsed["patient_id"],
                            "filename":     img_file.name,
                        })

    return records, skipped


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(records: list[dict], skipped: list[str]) -> None:
    """Print a human-readable summary to stdout."""
    n_total = len(records)
    print(f"\n{'='*60}")
    print(f"  BreaKHis Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total images   : {n_total}")
    print(f"  Skipped files  : {len(skipped)}")

    if not records:
        print("  (no records found — check data_dir)")
        return

    # ── By binary class ──────────────────────────────────────────────────────
    by_class: dict[str, int] = defaultdict(int)
    for r in records:
        by_class[r["binary_label"]] += 1
    print(f"\n  By binary class:")
    for cls in sorted(by_class):
        print(f"    {cls:>12s} : {by_class[cls]:>5d}")

    # ── By subtype ───────────────────────────────────────────────────────────
    by_subtype: dict[str, int] = defaultdict(int)
    for r in records:
        by_subtype[r["subtype"]] += 1
    print(f"\n  By subtype:")
    for subtype in sorted(by_subtype):
        marker = "(B)" if subtype in BENIGN_SUBTYPES else "(M)"
        print(f"    {subtype:>25s} {marker} : {by_subtype[subtype]:>5d}")

    # ── By magnification ────────────────────────────────────────────────────
    by_mag: dict[str, int] = defaultdict(int)
    for r in records:
        by_mag[r["magnification"]] += 1
    print(f"\n  By magnification:")
    for mag in sorted(VALID_MAGNIFICATIONS, key=lambda m: int(m[:-1])):
        print(f"    {mag:>5s} : {by_mag.get(mag, 0):>5d}")

    # ── By subtype × magnification ───────────────────────────────────────────
    by_subtype_mag: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        by_subtype_mag[(r["subtype"], r["magnification"])] += 1
    print(f"\n  By subtype × magnification:")
    mag_order = sorted(VALID_MAGNIFICATIONS, key=lambda m: int(m[:-1]))
    header = f"  {'subtype':>25s}  " + "  ".join(f"{m:>5s}" for m in mag_order)
    print(header)
    for subtype in sorted(by_subtype):
        row = f"  {subtype:>25s}  "
        row += "  ".join(
            f"{by_subtype_mag.get((subtype, m), 0):>5d}" for m in mag_order
        )
        print(row)

    # ── Unique patients ──────────────────────────────────────────────────────
    patients_by_subtype: dict[str, set[str]] = defaultdict(set)
    for r in records:
        patients_by_subtype[r["subtype"]].add(r["patient_id"])
    n_patients_total = len({r["patient_id"] for r in records})
    print(f"\n  Unique patients total : {n_patients_total}")
    print(f"  Patients by subtype:")
    for subtype in sorted(patients_by_subtype):
        print(f"    {subtype:>25s} : {len(patients_by_subtype[subtype]):>3d}")

    # ── Min samples per class (episode feasibility check) ───────────────────
    min_subtype = min(by_subtype.values()) if by_subtype else 0
    min_binary  = min(by_class.values()) if by_class else 0
    print(f"\n  Min images per subtype : {min_subtype}")
    print(f"  Min images per binary  : {min_binary}")
    print(f"  (Need >= k_shot + q_query per class per magnification for episodes)")

    print(f"{'='*60}\n")

    if skipped:
        print(f"  WARNING: {len(skipped)} files skipped (unrecognised filenames):")
        for path in skipped[:20]:
            print(f"    {path}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build BreakHis metadata CSV and print dataset statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Root of BreaKHis_v1 download (contains histology_slides/).",
    )
    p.add_argument(
        "--output_csv", default="data/breakhis_metadata.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--stats_only", action="store_true",
        help="Print statistics only; do not write CSV.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        logger.error(f"data_dir does not exist: {data_dir}")
        sys.exit(1)

    logger.info(f"Crawling: {data_dir}")
    records, skipped = crawl(data_dir)
    logger.info(f"Found {len(records)} images ({len(skipped)} skipped)")

    print_stats(records, skipped)

    if args.stats_only:
        return

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Metadata CSV written to: {output_csv}")


if __name__ == "__main__":
    main()
