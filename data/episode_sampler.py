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
