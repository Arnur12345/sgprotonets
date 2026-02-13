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


class IUXRayMultiLabelDataset(Dataset):
    """Indiana University Chest X-Ray dataset with multi-label support.

    Expects preprocessed data directory with:
        - metadata.csv: columns [image_path, report, labels]
          where labels is pipe-delimited: "effusion|cardiomegaly"
        - images/ subdirectory with image files

    This dataset provides:
        - Multi-hot encoded labels tensor
        - positive_indices[c]: Sample indices where class c IS present
        - negative_indices[c]: Sample indices where class c is ABSENT

    Args:
        data_dir: Path to preprocessed dataset root.
        split_classes: List of ALL possible class names to include.
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

        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        df = pd.read_csv(metadata_path)

        # Parse multi-label format: "effusion|cardiomegaly" -> ["effusion", "cardiomegaly"]
        if "labels" in df.columns:
            df["label_list"] = df["labels"].apply(
                lambda x: x.split("|") if pd.notna(x) and "|" in str(x) else [str(x)]
            )
        elif "label" in df.columns:
            # Fallback for single-label format
            df["label_list"] = df["label"].apply(lambda x: [str(x)])
        else:
            raise ValueError("metadata.csv must have 'labels' or 'label' column")

        # Filter to samples that have at least one of the split_classes
        split_classes_set = set(split_classes)
        mask = df["label_list"].apply(
            lambda labels: bool(split_classes_set.intersection(labels))
        )
        df = df[mask].reset_index(drop=True)

        self.image_paths: list[str] = df["image_path"].tolist()
        self.reports: list[str] = [
            preprocess_report(r) for r in df["report"].fillna("").tolist()
        ]
        self.label_lists: list[list[str]] = df["label_list"].tolist()

        # Build class-to-index mapping (sorted for reproducibility)
        self.classes = sorted(split_classes)
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Build multi-hot targets tensor
        self.targets: torch.Tensor = self._build_multihot()

        # Build binary indices for episode sampling
        self.positive_indices: dict[int, list[int]] = {}
        self.negative_indices: dict[int, list[int]] = {}
        self._build_binary_indices()

        # Also provide class_indices for compatibility with existing code
        self.class_indices: dict[int, list[int]] = self.positive_indices

    def _build_multihot(self) -> torch.Tensor:
        """Build multi-hot encoded targets tensor."""
        targets = torch.zeros(len(self.label_lists), self.num_classes)
        for idx, label_list in enumerate(self.label_lists):
            for lbl in label_list:
                if lbl in self.class_to_idx:
                    targets[idx, self.class_to_idx[lbl]] = 1.0
        return targets

    def _build_binary_indices(self) -> None:
        """Build positive and negative indices for each class."""
        for c in range(self.num_classes):
            pos_mask = self.targets[:, c] == 1
            neg_mask = self.targets[:, c] == 0

            self.positive_indices[c] = pos_mask.nonzero(as_tuple=True)[0].tolist()
            self.negative_indices[c] = neg_mask.nonzero(as_tuple=True)[0].tolist()

    def get_class_sample_counts(self) -> dict[str, dict[str, int]]:
        """Get positive and negative sample counts per class.

        Returns:
            Dict mapping class_name -> {"positive": count, "negative": count}
        """
        counts = {}
        for c in range(self.num_classes):
            class_name = self.classes[c]
            counts[class_name] = {
                "positive": len(self.positive_indices[c]),
                "negative": len(self.negative_indices[c]),
            }
        return counts

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | list[str]]:
        """Return a single sample.

        Returns:
            Dictionary with keys:
                - image: Tensor (3, H, W)
                - report: str
                - label: Tensor (num_classes,) multi-hot encoded
                - label_list: list[str] original label strings
        """
        img_path = self.data_dir / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "report": self.reports[idx],
            "label": self.targets[idx],  # Multi-hot tensor
            "label_list": self.label_lists[idx],  # Original string list
        }
