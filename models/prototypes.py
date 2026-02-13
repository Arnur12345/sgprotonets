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


def get_adaptive_anchor_weight(num_positives: int) -> float:
    """Compute anchor blending weight based on number of positive samples.

    More samples → trust visual prototype more (lower weight)
    Fewer samples → rely on semantic anchor more (higher weight)

    Args:
        num_positives: Number of positive samples for this label.

    Returns:
        Anchor weight α ∈ [0, 1], where final prototype = (1-α)*visual + α*anchor
    """
    if num_positives > 20:
        return 0.3
    elif num_positives > 10:
        return 0.5
    elif num_positives > 5:
        return 0.7
    else:
        return 0.9


class DualPrototypeComputation(nn.Module):
    """Compute positive and negative prototypes for binary decomposition.

    For each label l:
        P⁺_l = (weighted) mean of features where label l IS present
        P⁻_l = mean of features where label l is ABSENT

    Supports adaptive semantic anchor blending for rare labels:
        P⁺_l = (1 - α) * visual_mean + α * semantic_anchor
        where α increases as sample count decreases.

    Args:
        mode: "vanilla" for simple mean, "semantic_weighted" for similarity-weighted.
        use_adaptive_anchor: If True, adaptively blend positive prototypes with
            semantic anchors based on sample count.
        fixed_anchor_weight: If use_adaptive_anchor is False, use this fixed weight.
    """

    def __init__(
        self,
        mode: str = "vanilla",
        use_adaptive_anchor: bool = True,
        fixed_anchor_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.use_adaptive_anchor = use_adaptive_anchor
        self.fixed_anchor_weight = fixed_anchor_weight

    def forward(
        self,
        support_pos_features: torch.Tensor,
        support_neg_features: torch.Tensor,
        n_labels: int,
        k_pos: int,
        k_neg: int,
        class_semantics: torch.Tensor | None = None,
        label_sample_counts: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dual prototypes.

        Args:
            support_pos_features: Shape (n_labels * k_pos, d_model).
            support_neg_features: Shape (n_labels * k_neg, d_model).
            n_labels: Number of labels in episode.
            k_pos: Positive support examples per label.
            k_neg: Negative support examples per label.
            class_semantics: Optional class-level semantic embeddings,
                shape (n_labels, d_model). Required for semantic blending.
            label_sample_counts: Optional list of total positive sample counts
                per label (for adaptive anchor weighting).

        Returns:
            Tuple of (prototypes_pos, prototypes_neg), each shape (n_labels, d_model).
        """
        d_model = support_pos_features.shape[-1]
        device = support_pos_features.device

        # Reshape to (n_labels, k_pos/k_neg, d_model)
        pos_reshaped = support_pos_features.view(n_labels, k_pos, d_model)
        neg_reshaped = support_neg_features.view(n_labels, k_neg, d_model)

        if self.mode == "semantic_weighted" and class_semantics is not None:
            prototypes_pos = self._semantic_weighted_pos(pos_reshaped, class_semantics)
        else:
            # Vanilla: simple mean
            prototypes_pos = pos_reshaped.mean(dim=1)  # (n_labels, d_model)

        # Negative prototypes: always simple mean (no semantic guidance)
        prototypes_neg = neg_reshaped.mean(dim=1)  # (n_labels, d_model)

        # Apply adaptive anchor blending for positive prototypes
        if class_semantics is not None:
            prototypes_pos = self._apply_anchor_blending(
                prototypes_pos, class_semantics, label_sample_counts
            )

        return prototypes_pos, prototypes_neg

    def _semantic_weighted_pos(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Semantic-weighted aggregation for positive prototypes.

        Args:
            features: (n_labels, k_pos, d_model)
            anchors: (n_labels, d_model)

        Returns:
            (n_labels, d_model)
        """
        n_labels, k, d = features.shape
        prototypes = torch.zeros(n_labels, d, device=features.device)

        for l in range(n_labels):
            class_feats = features[l]  # (k, d)
            anchor = anchors[l]  # (d,)

            # Cosine similarity weights
            sim = F.cosine_similarity(class_feats, anchor.unsqueeze(0), dim=-1)
            weights = F.softmax(sim, dim=0)
            prototypes[l] = (weights.unsqueeze(-1) * class_feats).sum(dim=0)

        return prototypes

    def _apply_anchor_blending(
        self,
        visual_prototypes: torch.Tensor,
        semantic_anchors: torch.Tensor,
        sample_counts: list[int] | None,
    ) -> torch.Tensor:
        """Blend visual prototypes with semantic anchors.

        Args:
            visual_prototypes: (n_labels, d_model)
            semantic_anchors: (n_labels, d_model)
            sample_counts: Optional list of positive sample counts per label.

        Returns:
            Blended prototypes (n_labels, d_model)
        """
        n_labels = visual_prototypes.shape[0]

        if self.use_adaptive_anchor and sample_counts is not None:
            # Adaptive: different weight per label based on sample count
            blended = torch.zeros_like(visual_prototypes)
            for l in range(n_labels):
                alpha = get_adaptive_anchor_weight(sample_counts[l])
                blended[l] = (1 - alpha) * visual_prototypes[l] + alpha * semantic_anchors[l]
            return blended
        else:
            # Fixed weight for all labels
            alpha = self.fixed_anchor_weight
            return (1 - alpha) * visual_prototypes + alpha * semantic_anchors


def compute_binary_logits(
    query_features: torch.Tensor,
    prototypes_pos: torch.Tensor,
    prototypes_neg: torch.Tensor,
    distance: str = "cosine",
) -> torch.Tensor:
    """Compute binary logits for multi-label classification.

    For each query q and label l:
        logit_l = -dist(q, P⁺_l) + dist(q, P⁻_l)

    Positive logit means closer to positive prototype than negative prototype.

    Args:
        query_features: (n_query, d_model)
        prototypes_pos: (n_labels, d_model)
        prototypes_neg: (n_labels, d_model)
        distance: "cosine" or "euclidean"

    Returns:
        Binary logits, shape (n_query, n_labels)
    """
    dist_to_pos = compute_distances(query_features, prototypes_pos, distance)
    dist_to_neg = compute_distances(query_features, prototypes_neg, distance)

    # Binary logits: closer to positive = positive logit
    # logit = -dist_pos + dist_neg = dist_neg - dist_pos
    logits = dist_to_neg - dist_to_pos

    return logits
