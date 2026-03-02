"""BreakHis Breast Cancer Histopathological dataset loader.

Expected dataset root (BreaKHis_v1 as downloaded):

    <data_dir>/
    └── histology_slides/
        └── breast/
            ├── benign/
            │   └── SOB/
            │       ├── adenosis/
            │       │   └── <patient_folder>/
            │       │       ├── 40X/  → SOB_B_A-...-40-001.png
            │       │       ├── 100X/
            │       │       ├── 200X/
            │       │       └── 400X/
            │       ├── fibroadenoma/
            │       ├── phyllodes_tumor/
            │       └── tubular_adenoma/
            └── malignant/
                └── SOB/
                    ├── ductal_carcinoma/
                    ├── lobular_carcinoma/
                    ├── mucinous_carcinoma/
                    └── papillary_carcinoma/

Filename format:
    SOB_<CLASS>_<TYPE>-<YEAR>-<SLIDE>-<MAG>-<SEQ>.png

    e.g.  SOB_B_TA-14-4659-40-001.png
          SOB_M_DC-14-2523-400-003.png

Fields:
    CLASS  : B (benign) or M (malignant)
    TYPE   : abbreviation — A, F, PT, TA  (benign)
                           DC, LC, MC, PC (malignant)
    YEAR   : 2-digit collection year
    SLIDE  : numeric slide identifier
    MAG    : optical magnification — 40, 100, 200, 400
    SEQ    : 3-digit image sequence number within the slide/magnification
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from data.preprocessing import get_image_transform, preprocess_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENIGN_SUBTYPES: frozenset[str] = frozenset({
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
})

MALIGNANT_SUBTYPES: frozenset[str] = frozenset({
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
})

ALL_SUBTYPES: frozenset[str] = BENIGN_SUBTYPES | MALIGNANT_SUBTYPES

VALID_MAGNIFICATIONS: frozenset[str] = frozenset({"40X", "100X", "200X", "400X"})

# Filename type abbreviation → canonical subtype folder name
ABBREV_TO_SUBTYPE: dict[str, str] = {
    "A":  "adenosis",
    "F":  "fibroadenoma",
    "PT": "phyllodes_tumor",
    "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma",
    "LC": "lobular_carcinoma",
    "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}

# Anchored on known magnification values so patient_id captures everything
# between the type abbreviation and the magnification field.
# Example: SOB_B_TA-14-4659-40-001.png
#   cls=B, abbrev=TA, patient=14-4659, mag=40, seq=001
_FILENAME_RE = re.compile(
    r"^SOB_(?P<cls>[BM])_(?P<abbrev>[A-Z]+)"
    r"-(?P<patient>.+)"
    r"-(?P<mag>40|100|200|400)"
    r"-(?P<seq>\d+)\.png$"
)

LabelMode = Literal["binary", "subtype"]


# ---------------------------------------------------------------------------
# Filename parser (public utility)
# ---------------------------------------------------------------------------

def parse_breakhis_filename(filename: str) -> dict[str, str] | None:
    """Parse a BreakHis image filename into its semantic components.

    Args:
        filename: Basename only, e.g. ``SOB_B_TA-14-4659-40-001.png``.

    Returns:
        Dict with keys ``cls``, ``subtype``, ``patient_id``,
        ``magnification``, ``sequence``.  Returns ``None`` when the
        filename does not match the expected pattern or contains an
        unrecognised type abbreviation.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    abbrev = m.group("abbrev")
    subtype = ABBREV_TO_SUBTYPE.get(abbrev)
    if subtype is None:
        return None
    return {
        "cls":          m.group("cls"),           # "B" or "M"
        "subtype":      subtype,                  # e.g. "tubular_adenoma"
        "patient_id":   m.group("patient"),       # e.g. "14-4659"
        "magnification": m.group("mag") + "X",   # e.g. "40X"
        "sequence":     m.group("seq"),           # e.g. "001"
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BreakHisDataset(Dataset):
    """BreakHis Breast Cancer Histopathological Image dataset.

    Crawls the standard BreaKHis_v1 folder tree to collect all images and
    derives labels from the directory structure.  Optionally loads
    VLM-synthesised pathology reports from a CSV to populate the semantic
    modality expected by SGProtoNet.  When no report is available for an
    image the field is left as an empty string; the model then falls back
    to ``class_anchors`` inference strategy.

    Args:
        data_dir:
            Root of the BreaKHis_v1 download — the folder that contains
            the ``histology_slides/`` subdirectory.
        split_classes:
            Class names to include.  Interpretation depends on
            ``label_mode``:

            * ``"binary"``  — any subset of ``{"benign", "malignant"}``
            * ``"subtype"`` — any subset of the 8 canonical subtype names
              (e.g. ``["ductal_carcinoma", "lobular_carcinoma"]``)
        label_mode:
            ``"binary"`` for 2-class (benign / malignant) episodes or
            ``"subtype"`` for 8-class fine-grained episodes.
        magnifications:
            Magnification levels to include.  ``None`` keeps all four
            (40X, 100X, 200X, 400X).  Pass e.g. ``["40X"]`` to restrict
            to a single magnification for cross-magnification experiments.
        reports_csv:
            Optional path to a CSV with columns ``image_path`` and
            ``report`` containing VLM-generated pathology reports.
            ``image_path`` values must be relative to ``data_dir`` and
            use forward slashes.  Missing rows fall back to ``""``.
        image_size:
            Target image resolution for transforms (default 224).
        is_train:
            Whether to apply training-time augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        split_classes: list[str],
        label_mode: LabelMode = "subtype",
        magnifications: list[str] | None = None,
        reports_csv: str | None = None,
        image_size: int = 224,
        is_train: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.label_mode = label_mode
        self.transform: Compose = get_image_transform(image_size, is_train)

        # Validate magnification filter
        mags = set(magnifications) if magnifications is not None else set(VALID_MAGNIFICATIONS)
        invalid = mags - VALID_MAGNIFICATIONS
        if invalid:
            raise ValueError(
                f"Unknown magnification(s): {invalid}. "
                f"Valid options: {sorted(VALID_MAGNIFICATIONS)}"
            )
        self._magnifications = mags

        # Load optional synthetic reports: relative_path → report string
        self._reports_lookup: dict[str, str] = {}
        if reports_csv is not None:
            self._reports_lookup = self._load_reports_csv(reports_csv)

        # Collect samples by crawling the folder tree
        samples = self._crawl(set(split_classes))
        if not samples:
            raise RuntimeError(
                f"No images found under '{self.data_dir}' for classes "
                f"{split_classes} at magnifications {sorted(mags)}. "
                f"Check that data_dir points to the BreaKHis_v1 root."
            )

        self.image_paths: list[str]       = [s["image_path"]   for s in samples]
        self.reports: list[str]           = [s["report"]        for s in samples]
        self.labels: list[str]            = [s["label"]         for s in samples]
        self.patient_ids: list[str]       = [s["patient_id"]    for s in samples]
        self.magnifications_meta: list[str] = [s["magnification"] for s in samples]

        # Sorted class list for reproducibility (matches iu_xray.py pattern)
        self.classes: list[str] = sorted(set(self.labels))
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.targets: list[int] = [self.class_to_idx[l] for l in self.labels]

        # class_indices required by EpisodeSampler
        self.class_indices: dict[int, list[int]] = {}
        for idx, target in enumerate(self.targets):
            self.class_indices.setdefault(target, []).append(idx)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_reports_csv(self, reports_csv: str) -> dict[str, str]:
        """Load VLM-generated report lookup from CSV."""
        df = pd.read_csv(reports_csv)
        required = {"image_path", "report"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"reports_csv must have columns {required}, got {list(df.columns)}"
            )
        return dict(zip(df["image_path"], df["report"].fillna("")))

    def _label_for(self, subtype: str, class_name: str) -> str:
        """Derive the label string from the class name and label mode."""
        if self.label_mode == "binary":
            return class_name  # "benign" or "malignant"
        return subtype         # e.g. "ductal_carcinoma"

    def _crawl(self, split_set: set[str]) -> list[dict]:
        """Walk the BreaKHis_v1 directory tree and collect image records.

        Traversal path:
            histology_slides/breast/<class_name>/SOB/<subtype>/<patient>/<mag>/*.png
        """
        slides_root = self.data_dir / "histology_slides" / "breast"
        samples: list[dict] = []

        for class_name in ("benign", "malignant"):
            sob_dir = slides_root / class_name / "SOB"
            if not sob_dir.is_dir():
                continue

            for subtype_dir in sorted(sob_dir.iterdir()):
                if not subtype_dir.is_dir():
                    continue

                subtype = subtype_dir.name          # e.g. "ductal_carcinoma"
                label = self._label_for(subtype, class_name)
                if label not in split_set:
                    continue

                # patient_folder → mag_folder → *.png
                for patient_dir in sorted(subtype_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue

                    for mag_dir in sorted(patient_dir.iterdir()):
                        if not mag_dir.is_dir():
                            continue
                        if mag_dir.name not in self._magnifications:
                            continue

                        for img_file in sorted(mag_dir.glob("*.png")):
                            parsed = parse_breakhis_filename(img_file.name)
                            if parsed is None:
                                continue

                            rel_path = str(img_file.relative_to(self.data_dir))
                            report = preprocess_report(
                                self._reports_lookup.get(rel_path, "")
                            )
                            samples.append({
                                "image_path":   rel_path,
                                "report":       report,
                                "label":        label,
                                "patient_id":   parsed["patient_id"],
                                "magnification": parsed["magnification"],
                            })
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        """Return a single sample.

        Returns:
            Dict with keys:

            * ``image``       — ``Tensor (3, H, W)``
            * ``report``      — ``str`` (empty when no synthetic report)
            * ``label``       — ``int`` class index
            * ``class_name``  — ``str`` label string
            * ``patient_id``  — ``str`` e.g. ``"14-4659"``
            * ``magnification`` — ``str`` e.g. ``"40X"``
        """
        img_path = self.data_dir / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image":         image,
            "report":        self.reports[idx],
            "label":         self.targets[idx],
            "class_name":    self.labels[idx],
            "patient_id":    self.patient_ids[idx],
            "magnification": self.magnifications_meta[idx],
        }
