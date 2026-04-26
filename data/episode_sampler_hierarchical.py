"""Hierarchical dual-mode episodic sampler for BreakHis.

Randomly alternates between two episode granularities in each epoch:

* **Binary mode** — 2-way episodes (benign vs malignant).  The sampler
  derives binary class membership from the subtype labels already present
  in a ``BreakHisDataset(label_mode="subtype")`` instance, so a single
  dataset object is enough.

* **Subtype mode** — N-way episodes using the 8 fine-grained subtypes.

One model trains on both episode types simultaneously.  Semantic anchors
act as magnification-invariant identifiers that allow the same learned
representations to handle coarse and fine granularity without retraining.

Example usage (iterate directly, as with BinaryEpisodeSampler)::

    from data.breakhis import BreakHisDataset, BENIGN_SUBTYPES
    from data.episode_sampler_hierarchical import HierarchicalEpisodeSampler

    dataset = BreakHisDataset(
        data_dir="data/BreaKHis_v1/",
        split_classes=list(BENIGN_SUBTYPES | MALIGNANT_SUBTYPES),
        label_mode="subtype",
        magnifications=["40X"],
    )

    sampler = HierarchicalEpisodeSampler(
        dataset=dataset,
        n_way_subtype=5,
        k_shot=5,
        q_query=8,
        num_episodes=300,
        binary_prob=0.3,   # 30 % binary, 70 % subtype
    )

    for episode_meta in sampler:
        indices    = episode_meta["indices"]
        mode       = episode_meta["mode"]       # "binary" | "subtype"
        n_way      = episode_meta["n_way"]
        k_shot     = episode_meta["k_shot"]
        q_query    = episode_meta["q_query"]

        samples  = [dataset[i] for i in indices]
        batch    = collate_episode(samples)
        support, query, query_labels = unpack_hierarchical_episode(batch, episode_meta)
"""

from __future__ import annotations

import random
from typing import Literal

import torch
from torch.utils.data import Dataset, Sampler

from data.breakhis import BENIGN_SUBTYPES, MALIGNANT_SUBTYPES

EpisodeMode = Literal["binary", "subtype"]


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_episode(samples: list[dict]) -> dict:
    """Stack a list of dataset samples into a batched dict.

    Handles both ``image`` tensors and variable-length ``report`` strings.
    Compatible with ``BreakHisDataset.__getitem__`` output format.
    """
    images  = torch.stack([s["image"] for s in samples])
    reports = [s["report"] for s in samples]
    labels  = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    return {"image": images, "report": reports, "label": labels}


# ---------------------------------------------------------------------------
# Unpack helper
# ---------------------------------------------------------------------------

def unpack_hierarchical_episode(
    batch: dict[str, torch.Tensor | list],
    episode_meta: dict,
) -> tuple[dict, dict, torch.Tensor]:
    """Split a hierarchical episode into support/query sets.

    Remaps labels to contiguous 0…N-1 indices within the episode,
    matching the convention of ``unpack_episode`` in episode_sampler.py.

    Args:
        batch:        Collated batch returned by ``collate_episode``.
        episode_meta: Dict yielded by ``HierarchicalEpisodeSampler``.

    Returns:
        ``(support_batch, query_batch, query_labels)`` where
        ``query_labels`` are integer class indices 0…N-1.
    """
    n_way   = episode_meta["n_way"]
    k_shot  = episode_meta["k_shot"]
    q_query = episode_meta["q_query"]

    n_support = n_way * k_shot
    n_query   = n_way * q_query

    support = {
        "image":  batch["image"][:n_support],
        "report": batch["report"][:n_support],
        "label":  torch.arange(n_way).repeat_interleave(k_shot),
    }
    query = {
        "image":  batch["image"][n_support:n_support + n_query],
        "report": batch["report"][n_support:n_support + n_query],
    }
    query_labels = torch.arange(n_way).repeat_interleave(q_query)
    query["label"] = query_labels

    return support, query, query_labels


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class HierarchicalEpisodeSampler(Sampler[dict]):
    """Stochastically alternates between binary and subtype episodes.

    Operates on a ``BreakHisDataset`` built with ``label_mode="subtype"``.
    Binary class membership is derived automatically from subtype labels
    using ``BENIGN_SUBTYPES`` / ``MALIGNANT_SUBTYPES`` constants.

    Args:
        dataset:
            A ``BreakHisDataset`` with ``label_mode="subtype"``.  Must
            expose ``class_indices`` (subtype int → list[sample_idx]) and
            ``classes`` (list of subtype name strings).
        n_way_subtype:
            Number of classes per subtype episode.  Must not exceed the
            number of available subtype classes in the dataset.
        k_shot:
            Support examples per class (shared by both modes).
        q_query:
            Query examples per class (shared by both modes).
        num_episodes:
            Total episodes per epoch (binary + subtype combined).
        binary_prob:
            Probability that each episode is sampled in binary mode.
            Set to 0.0 for pure subtype training; 1.0 for pure binary.
        seed:
            Random seed (incremented each epoch for non-repeating schedules).

    Raises:
        ValueError:
            If the dataset does not have ``label_mode="subtype"``, if
            ``n_way_subtype`` exceeds available classes, or if any class
            lacks sufficient samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way_subtype: int,
        k_shot: int,
        q_query: int,
        num_episodes: int,
        binary_prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        if getattr(dataset, "label_mode", None) != "subtype":
            raise ValueError(
                "HierarchicalEpisodeSampler requires a BreakHisDataset with "
                "label_mode='subtype'. Received label_mode="
                f"'{getattr(dataset, 'label_mode', 'unknown')}'."
            )

        self.k_shot       = k_shot
        self.q_query      = q_query
        self.num_episodes = num_episodes
        self.binary_prob  = binary_prob
        self.seed         = seed
        self._epoch       = 0

        needed = k_shot + q_query

        # ── Subtype class indices ────────────────────────────────────────────
        self.subtype_class_indices: dict[int, list[int]] = dataset.class_indices
        self.subtype_classes: list[str] = dataset.classes  # sorted subtype names

        if n_way_subtype > len(self.subtype_classes):
            raise ValueError(
                f"n_way_subtype={n_way_subtype} exceeds available subtype classes "
                f"({len(self.subtype_classes)}): {self.subtype_classes}"
            )
        self.n_way_subtype = n_way_subtype

        # Validate each subtype class has enough samples
        for cls_idx, indices in self.subtype_class_indices.items():
            if len(indices) < needed:
                raise ValueError(
                    f"Subtype class index {cls_idx} "
                    f"('{self.subtype_classes[cls_idx]}') has {len(indices)} "
                    f"samples, need at least {needed}."
                )

        # ── Binary class indices derived from subtype labels ─────────────────
        # Map sample position → binary class (0=benign, 1=malignant)
        binary_indices: dict[int, list[int]] = {0: [], 1: []}
        for cls_idx, cls_name in enumerate(self.subtype_classes):
            if cls_name in BENIGN_SUBTYPES:
                binary_indices[0].extend(self.subtype_class_indices[cls_idx])
            elif cls_name in MALIGNANT_SUBTYPES:
                binary_indices[1].extend(self.subtype_class_indices[cls_idx])
        self.binary_class_indices = binary_indices

        for cls_idx, indices in self.binary_class_indices.items():
            label = "benign" if cls_idx == 0 else "malignant"
            if len(indices) < needed:
                raise ValueError(
                    f"Binary class '{label}' has {len(indices)} samples, "
                    f"need at least {needed}."
                )

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        """Yield episode metadata dicts for each episode in the epoch.

        Each dict contains:

        * ``indices``       — flat list of dataset sample indices
          (support first, then query; class-ordered within each block)
        * ``mode``          — ``"binary"`` or ``"subtype"``
        * ``n_way``         — number of classes in this episode
        * ``k_shot``        — support examples per class
        * ``q_query``       — query examples per class
        * ``episode_classes`` — list of *local* class indices for this
          episode (0-indexed within the episode).  For binary episodes
          this is always ``[0, 1]``; for subtype episodes it is a sample
          from the available subtype pool.
        * ``class_names``   — list of class name strings for each episode
          class, useful for retrieving class anchors by name.
        """
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        subtype_pool = list(self.subtype_class_indices.keys())

        for _ in range(self.num_episodes):
            if rng.random() < self.binary_prob:
                yield self._sample_binary_episode(rng)
            else:
                yield self._sample_subtype_episode(rng, subtype_pool)

    def __len__(self) -> int:
        return self.num_episodes

    # ------------------------------------------------------------------
    # Internal episode constructors
    # ------------------------------------------------------------------

    def _sample_binary_episode(self, rng: random.Random) -> dict:
        """Sample a 2-way binary (benign / malignant) episode."""
        support_indices: list[int] = []
        query_indices:   list[int] = []
        class_names:     list[str] = ["benign", "malignant"]

        for cls_idx in (0, 1):
            pool     = self.binary_class_indices[cls_idx]
            selected = rng.sample(pool, self.k_shot + self.q_query)
            support_indices.extend(selected[: self.k_shot])
            query_indices.extend(selected[self.k_shot :])

        return {
            "indices":         support_indices + query_indices,
            "mode":            "binary",
            "n_way":           2,
            "k_shot":          self.k_shot,
            "q_query":         self.q_query,
            "episode_classes": [0, 1],
            "class_names":     class_names,
        }

    def _sample_subtype_episode(
        self, rng: random.Random, subtype_pool: list[int]
    ) -> dict:
        """Sample an n_way_subtype fine-grained subtype episode."""
        episode_classes = rng.sample(subtype_pool, self.n_way_subtype)
        support_indices: list[int] = []
        query_indices:   list[int] = []

        for cls_idx in episode_classes:
            pool     = self.subtype_class_indices[cls_idx]
            selected = rng.sample(pool, self.k_shot + self.q_query)
            support_indices.extend(selected[: self.k_shot])
            query_indices.extend(selected[self.k_shot :])

        class_names = [self.subtype_classes[c] for c in episode_classes]
        return {
            "indices":         support_indices + query_indices,
            "mode":            "subtype",
            "n_way":           self.n_way_subtype,
            "k_shot":          self.k_shot,
            "q_query":         self.q_query,
            "episode_classes": list(range(self.n_way_subtype)),
            "class_names":     class_names,
        }
