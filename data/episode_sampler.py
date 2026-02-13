"""N-way K-shot episodic sampler for meta-learning."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, Sampler


class EpisodeSampler(Sampler[List[int]]):
    """Samples N-way K-shot episodes from a dataset.

    Each episode consists of:
        - N classes sampled from available classes
        - K support examples per class
        - Q query examples per class

    Args:
        dataset: Dataset with class_indices and classes attributes.
        n_way: Number of classes per episode.
        k_shot: Number of support examples per class.
        q_query: Number of query examples per class.
        num_episodes: Number of episodes per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        k_shot: int,
        q_query: int,
        num_episodes: int,
        seed: int = 42,
    ) -> None:
        self.class_indices: dict[int, list[int]] = dataset.class_indices
        self.num_classes: int = len(dataset.classes)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.seed = seed

        # Validate that each class has enough samples
        for cls_idx, indices in self.class_indices.items():
            if len(indices) < k_shot + q_query:
                raise ValueError(
                    f"Class {cls_idx} has {len(indices)} samples, "
                    f"need at least {k_shot + q_query} (k_shot + q_query)"
                )

    def __iter__(self):
        """Yield episode index batches.

        Each batch is a flat list of indices:
            [support_cls0_0, ..., support_cls0_K, ..., support_clsN_K,
             query_cls0_0, ..., query_cls0_Q, ..., query_clsN_Q]
        """
        rng = random.Random(self.seed)
        class_pool = list(self.class_indices.keys())

        for _ in range(self.num_episodes):
            # Sample N classes
            episode_classes = rng.sample(class_pool, self.n_way)

            support_indices = []
            query_indices = []

            for cls_idx in episode_classes:
                available = self.class_indices[cls_idx]
                selected = rng.sample(available, self.k_shot + self.q_query)
                support_indices.extend(selected[: self.k_shot])
                query_indices.extend(selected[self.k_shot :])

            yield support_indices + query_indices

    def __len__(self) -> int:
        return self.num_episodes


def unpack_episode(
    batch: dict[str, torch.Tensor | list],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Split a flat episode batch into support set, query set, and query labels.

    Args:
        batch: Collated batch from DataLoader with episode sampler.
        n_way: Number of classes.
        k_shot: Support examples per class.
        q_query: Query examples per class.

    Returns:
        Tuple of (support_batch, query_batch, query_labels).
        query_labels are class indices 0..N-1 within the episode.
    """
    n_support = n_way * k_shot
    n_query = n_way * q_query

    support = {
        "image": batch["image"][:n_support],
        "report": batch["report"][:n_support],
        "label": batch["label"][:n_support],
    }
    query = {
        "image": batch["image"][n_support:],
        "report": batch["report"][n_support:],
        "label": batch["label"][n_support:],
    }

    # Remap labels to 0..N-1 within the episode
    # Support labels are ordered: k_shot copies of class 0, k_shot of class 1, etc.
    support_episode_labels = torch.arange(n_way).repeat_interleave(k_shot)
    support["label"] = support_episode_labels

    # Query labels: q_query copies of class 0, q_query of class 1, etc.
    query_episode_labels = torch.arange(n_way).repeat_interleave(q_query)
    query["label"] = query_episode_labels

    return support, query, query_episode_labels


class BinaryEpisodeSampler(Sampler[Dict]):
    """Samples episodes for binary decomposition multi-label classification.

    Each episode samples L labels. For each label l:
        - k_pos positive support samples (label present)
        - k_neg negative support samples (label absent)
        - q_query query samples (balanced mix of positive/negative)

    Episode structure:
        [support_pos (L × k_pos), support_neg (L × k_neg), query (L × q_query)]

    Note: The same image can appear as positive for label A and negative for
    label B within the same episode - this is intended multi-label behavior.

    Args:
        dataset: MultiLabelDataset with positive_indices and negative_indices.
        n_labels: Number of labels per episode (L).
        k_pos: Positive support examples per label.
        k_neg: Negative support examples per label.
        q_query: Query examples per label (will be balanced pos/neg).
        num_episodes: Number of episodes per epoch.
        min_positives: Minimum positive samples required to include a label.
        min_negatives: Minimum negative samples required.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_labels: int,
        k_pos: int,
        k_neg: int,
        q_query: int,
        num_episodes: int,
        min_positives: int = 3,
        min_negatives: int = 10,
        seed: int = 42,
    ) -> None:
        self.positive_indices: dict[int, list[int]] = dataset.positive_indices
        self.negative_indices: dict[int, list[int]] = dataset.negative_indices
        self.num_classes: int = dataset.num_classes
        self.classes: list[str] = dataset.classes
        self.n_labels = n_labels
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.seed = seed

        # Filter to labels with sufficient samples for support + query
        n_pos_query = q_query // 2
        n_neg_query = q_query - n_pos_query

        self.valid_labels: list[int] = []
        for c in range(self.num_classes):
            n_pos = len(self.positive_indices[c])
            n_neg = len(self.negative_indices[c])

            has_enough_pos = n_pos >= max(min_positives, k_pos + n_pos_query)
            has_enough_neg = n_neg >= max(min_negatives, k_neg + n_neg_query)

            if has_enough_pos and has_enough_neg:
                self.valid_labels.append(c)

        if len(self.valid_labels) < n_labels:
            raise ValueError(
                f"Only {len(self.valid_labels)} labels have sufficient samples, "
                f"need at least {n_labels}. Valid labels: "
                f"{[self.classes[i] for i in self.valid_labels]}"
            )

    def __iter__(self):
        """Yield episode batches.

        Each batch is a dict containing:
            - indices: Flat list [support_pos, support_neg, query]
            - episode_labels: Which global label indices are in this episode
            - n_labels, k_pos, k_neg, q_query: Episode structure params
        """
        rng = random.Random(self.seed)

        for _ in range(self.num_episodes):
            # Sample L labels for this episode
            episode_labels = rng.sample(self.valid_labels, self.n_labels)

            support_pos_indices: list[int] = []
            support_neg_indices: list[int] = []
            query_indices: list[int] = []

            n_pos_query = self.q_query // 2
            n_neg_query = self.q_query - n_pos_query

            for label_idx in episode_labels:
                # Get and shuffle available indices
                pos_pool = self.positive_indices[label_idx].copy()
                neg_pool = self.negative_indices[label_idx].copy()
                rng.shuffle(pos_pool)
                rng.shuffle(neg_pool)

                # Support: k_pos positives, k_neg negatives
                support_pos_indices.extend(pos_pool[: self.k_pos])
                support_neg_indices.extend(neg_pool[: self.k_neg])

                # Query: balanced mix of positive and negative
                query_pos = pos_pool[self.k_pos : self.k_pos + n_pos_query]
                query_neg = neg_pool[self.k_neg : self.k_neg + n_neg_query]
                query_indices.extend(query_pos + query_neg)

            # Yield dict with indices and metadata
            yield {
                "indices": support_pos_indices + support_neg_indices + query_indices,
                "episode_labels": episode_labels,
                "n_labels": self.n_labels,
                "k_pos": self.k_pos,
                "k_neg": self.k_neg,
                "q_query": self.q_query,
            }

    def __len__(self) -> int:
        return self.num_episodes


def unpack_binary_episode(
    batch: dict[str, torch.Tensor | list],
    episode_labels: list[int],
    n_labels: int,
    k_pos: int,
    k_neg: int,
    q_query: int,
) -> tuple[dict, dict, dict, torch.Tensor, list[int]]:
    """Unpack binary decomposition episode into support_pos, support_neg, query.

    Args:
        batch: Collated batch from DataLoader with BinaryEpisodeSampler.
        episode_labels: Which global class indices are in this episode.
        n_labels: Number of labels in episode.
        k_pos: Positive support examples per label.
        k_neg: Negative support examples per label.
        q_query: Query examples per label.

    Returns:
        Tuple of (support_pos, support_neg, query, query_labels, episode_labels):
            - support_pos: Dict with image/report for positive support (n_labels * k_pos)
            - support_neg: Dict with image/report for negative support (n_labels * k_neg)
            - query: Dict with image/report for queries (n_labels * q_query)
            - query_labels: Multi-hot tensor (n_query, n_labels) for episode labels
            - episode_labels: Original global label indices
    """
    n_support_pos = n_labels * k_pos
    n_support_neg = n_labels * k_neg
    n_query = n_labels * q_query

    images = batch["image"]
    reports = batch["report"]
    global_labels = batch["label"]  # Multi-hot from dataset (n_total, num_classes)

    # Split by position
    support_pos = {
        "image": images[:n_support_pos],
        "report": reports[:n_support_pos] if isinstance(reports, torch.Tensor) else reports[:n_support_pos],
    }
    support_neg = {
        "image": images[n_support_pos : n_support_pos + n_support_neg],
        "report": reports[n_support_pos : n_support_pos + n_support_neg]
        if isinstance(reports, torch.Tensor)
        else reports[n_support_pos : n_support_pos + n_support_neg],
    }
    query = {
        "image": images[n_support_pos + n_support_neg :],
        "report": reports[n_support_pos + n_support_neg :]
        if isinstance(reports, torch.Tensor)
        else reports[n_support_pos + n_support_neg :],
    }

    # Extract query labels for episode labels only (multi-hot subset)
    query_global_labels = global_labels[n_support_pos + n_support_neg :]  # (n_query, num_classes)

    # Convert episode_labels to tensor for indexing
    episode_labels_tensor = torch.tensor(episode_labels, dtype=torch.long)
    query_labels = query_global_labels[:, episode_labels_tensor]  # (n_query, n_labels)

    return support_pos, support_neg, query, query_labels, episode_labels


def binary_episode_collate_fn(batch_list: list[dict]) -> dict:
    """Custom collate function for BinaryEpisodeSampler.

    Since BinaryEpisodeSampler yields dicts with 'indices' key,
    the DataLoader needs special handling.

    Args:
        batch_list: List of sample dicts from dataset.__getitem__

    Returns:
        Collated batch dict with stacked tensors.
    """
    images = torch.stack([b["image"] for b in batch_list])
    reports = [b["report"] for b in batch_list]

    # Handle both multi-hot (tensor) and single-label (int) formats
    if isinstance(batch_list[0]["label"], torch.Tensor):
        labels = torch.stack([b["label"] for b in batch_list])
    else:
        labels = torch.tensor([b["label"] for b in batch_list])

    return {
        "image": images,
        "report": reports,
        "label": labels,
    }
