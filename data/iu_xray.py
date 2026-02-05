"""IU Chest X-Ray dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from data.preprocessing import preprocess_report, get_image_transform


class IUXRayDataset(Dataset):
    """Indiana University Chest X-Ray dataset.

    Expects preprocessed data directory with:
        - metadata.csv: columns [image_path, report, label]
        - images/ subdirectory with image files

    Args:
        data_dir: Path to preprocessed dataset root.
        split_classes: List of class names to include.
        image_size: Target image resolution.
        is_train: Whether to use training augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        split_classes: list[str],
        image_size: int = 224,
        is_train: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform: Compose = get_image_transform(image_size, is_train)

        # Load metadata and filter to requested classes
        metadata_path = self.data_dir / "metadata.csv"
        df = pd.read_csv(metadata_path)
        df = df[df["label"].isin(split_classes)].reset_index(drop=True)

        self.image_paths: list[str] = df["image_path"].tolist()
        self.reports: list[str] = [
            preprocess_report(r) for r in df["report"].fillna("").tolist()
        ]
        self.labels: list[str] = df["label"].tolist()

        # Build class-to-index mapping (sorted for reproducibility)
        self.classes = sorted(set(self.labels))
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.targets: list[int] = [self.class_to_idx[l] for l in self.labels]

        # Build index lookup per class for episode sampling
        self.class_indices: dict[int, list[int]] = {}
        for idx, target in enumerate(self.targets):
            self.class_indices.setdefault(target, []).append(idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        """Return a single sample.

        Returns:
            Dictionary with keys: image, report, label (int), class_name (str).
        """
        img_path = self.data_dir / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "report": self.reports[idx],
            "label": self.targets[idx],
            "class_name": self.labels[idx],
        }
