"""Prototype computation for few-shot classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeComputation(nn.Module):
    """Compute class prototypes from support set embeddings.

    Supports vanilla (mean) and semantic-weighted aggregation modes.

    Args:
        mode: "vanilla" for simple mean, "semantic_weighted" for similarity-weighted.
    """

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()
        self.mode = mode

    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        class_semantics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute prototypes.

        Args:
            support_features: Shape (n_way * k_shot, d_model).
            support_labels: Shape (n_way * k_shot,), values in 0..n_way-1.
            n_way: Number of classes.
            class_semantics: Optional class-level semantic embeddings,
                shape (n_way, d_model). Required for semantic_weighted mode.

        Returns:
            Prototypes, shape (n_way, d_model).
        """
        if self.mode == "semantic_weighted" and class_semantics is not None:
            return self._semantic_weighted(
                support_features, support_labels, n_way, class_semantics
            )
        return self._vanilla(support_features, support_labels, n_way)

    def _vanilla(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Simple mean prototype computation.

        Args:
            support_features: (n_way * k_shot, d_model).
            support_labels: (n_way * k_shot,).
            n_way: Number of classes.

        Returns:
            (n_way, d_model).
        """
        d_model = support_features.shape[-1]
        prototypes = torch.zeros(n_way, d_model, device=support_features.device)
        for c in range(n_way):
            mask = support_labels == c
            prototypes[c] = support_features[mask].mean(dim=0)
        return prototypes

    def _semantic_weighted(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
        class_semantics: torch.Tensor,
    ) -> torch.Tensor:
        """Semantic-weighted prototype aggregation.

        Weights each support example by its similarity to the class-level
        semantic description.

        Args:
            support_features: (n_way * k_shot, d_model).
            support_labels: (n_way * k_shot,).
            n_way: Number of classes.
            class_semantics: (n_way, d_model).

        Returns:
            (n_way, d_model).
        """
        d_model = support_features.shape[-1]
        prototypes = torch.zeros(n_way, d_model, device=support_features.device)

        for c in range(n_way):
            mask = support_labels == c
            class_feats = support_features[mask]  # (k_shot, d_model)
            anchor = class_semantics[c]  # (d_model,)

            # Cosine similarity weights
            sim = F.cosine_similarity(class_feats, anchor.unsqueeze(0), dim=-1)  # (k_shot,)
            weights = F.softmax(sim, dim=0)  # (k_shot,)
            prototypes[c] = (weights.unsqueeze(-1) * class_feats).sum(dim=0)

        return prototypes


def compute_distances(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
    distance: str = "cosine",
) -> torch.Tensor:
    """Compute distances between query features and prototypes.

    Args:
        query_features: (n_query, d_model).
        prototypes: (n_way, d_model).
        distance: "cosine" or "euclidean".

    Returns:
        Distance matrix, shape (n_query, n_way). Lower = more similar.
    """
    if distance == "cosine":
        # L2-normalize before cosine distance
        q_norm = F.normalize(query_features, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        # Cosine distance = 1 - cosine_similarity
        return 1.0 - torch.mm(q_norm, p_norm.t())
    elif distance == "euclidean":
        # Squared Euclidean distance
        return torch.cdist(query_features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
    else:
        raise ValueError(f"Unknown distance: {distance}")
