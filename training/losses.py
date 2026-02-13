"""Loss functions: L_proto, L_align (InfoNCE), L_consist."""

import torch
import torch.nn.functional as F


def prototypical_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Prototypical cross-entropy loss over query predictions.

    Args:
        logits: (n_query, n_way), higher = more similar.
        labels: (n_query,), ground-truth class indices 0..n_way-1.

    Returns:
        Scalar loss.
    """
    return F.cross_entropy(logits, labels)


def alignment_loss(
    v_proj: torch.Tensor,
    s_proj: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss between visual and semantic embeddings.

    Symmetric: image-to-text + text-to-image.

    Args:
        v_proj: Projected visual features, shape (B, d_model).
        s_proj: Projected semantic features, shape (B, d_model).
        temperature: Softmax temperature.

    Returns:
        Scalar loss.
    """
    # L2-normalize
    v_norm = F.normalize(v_proj, p=2, dim=-1)
    s_norm = F.normalize(s_proj, p=2, dim=-1)

    # Similarity matrix: (B, B)
    sim = torch.mm(v_norm, s_norm.t()) / temperature

    # Labels: diagonal (matched pairs)
    labels = torch.arange(sim.shape[0], device=sim.device)

    # Symmetric InfoNCE
    loss_v2s = F.cross_entropy(sim, labels)
    loss_s2v = F.cross_entropy(sim.t(), labels)

    return (loss_v2s + loss_s2v) / 2


def consistency_loss(
    f_final: torch.Tensor,
    v_cls_proj: torch.Tensor,
) -> torch.Tensor:
    """Consistency loss between fused representation and visual-only features.

    Encourages the fused representation to stay close to the visual features,
    preventing text from dominating.

    Args:
        f_final: Fused multimodal features, shape (B, d_model).
        v_cls_proj: Projected visual [CLS] features, shape (B, d_model).

    Returns:
        Scalar loss (MSE).
    """
    return F.mse_loss(
        F.normalize(f_final, p=2, dim=-1),
        F.normalize(v_cls_proj, p=2, dim=-1),
    )


def vis2sem_loss(
    predicted_s: torch.Tensor,
    target_s: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for visual-to-semantic prediction.

    Args:
        predicted_s: Predicted semantic embeddings, shape (B, d_model).
        target_s: Target semantic embeddings (from actual text), shape (B, d_model).

    Returns:
        Scalar loss.
    """
    return F.mse_loss(predicted_s, target_s.detach())


# =============================================================================
# Multi-Label Loss Functions (Binary Decomposition)
# =============================================================================


def multilabel_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy loss for multi-label classification.

    Args:
        logits: Raw logits (before sigmoid), shape (n_query, n_labels).
        labels: Multi-hot ground truth (0/1), shape (n_query, n_labels).
        pos_weight: Per-label weight for positive class to handle imbalance,
            shape (n_labels,). Optional.

    Returns:
        Scalar loss.
    """
    return F.binary_cross_entropy_with_logits(
        logits, labels.float(), pos_weight=pos_weight
    )


def focal_multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Focal loss for multi-label classification with extreme label imbalance.

    Focal loss down-weights easy examples and focuses learning on hard ones.
    This is particularly useful when some labels are very rare.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        logits: Raw logits (before sigmoid), shape (n_query, n_labels).
        labels: Multi-hot ground truth (0/1), shape (n_query, n_labels).
        gamma: Focusing parameter. γ=0 recovers standard BCE.
            Higher γ increases focus on hard examples.
        alpha: Balance parameter for positive/negative. α=0.25 is common.

    Returns:
        Scalar loss.
    """
    # Compute probabilities
    probs = torch.sigmoid(logits)

    # Binary cross-entropy (element-wise, no reduction)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )

    # p_t: probability of true class
    # For positive labels (y=1): p_t = probs
    # For negative labels (y=0): p_t = 1 - probs
    p_t = probs * labels + (1 - probs) * (1 - labels)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting for class imbalance
    # For positive labels: use alpha
    # For negative labels: use (1 - alpha)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)

    # Final focal loss
    loss = alpha_t * focal_weight * bce_loss

    return loss.mean()


def prototype_margin_loss(
    query_features: torch.Tensor,
    prototypes_pos: torch.Tensor,
    prototypes_neg: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Triplet-style margin loss for better prototype separation.

    Encourages:
        - For positive labels: dist(q, P+) + margin < dist(q, P-)
        - For negative labels: dist(q, P-) + margin < dist(q, P+)

    This helps create clearer decision boundaries between positive and
    negative prototypes.

    Args:
        query_features: Query embeddings, shape (n_query, d_model).
        prototypes_pos: Positive prototypes, shape (n_labels, d_model).
        prototypes_neg: Negative prototypes, shape (n_labels, d_model).
        labels: Multi-hot ground truth, shape (n_query, n_labels).
        margin: Margin for triplet loss.

    Returns:
        Scalar loss.
    """
    # Compute distances
    # L2-normalize for cosine-like distance
    q_norm = F.normalize(query_features, p=2, dim=-1)
    pos_norm = F.normalize(prototypes_pos, p=2, dim=-1)
    neg_norm = F.normalize(prototypes_neg, p=2, dim=-1)

    # Cosine distance = 1 - cosine_similarity
    dist_to_pos = 1.0 - torch.mm(q_norm, pos_norm.t())  # (n_query, n_labels)
    dist_to_neg = 1.0 - torch.mm(q_norm, neg_norm.t())  # (n_query, n_labels)

    # For positive labels: want dist_pos < dist_neg - margin
    # Violation: max(0, dist_pos - dist_neg + margin)
    pos_mask = labels.bool()
    pos_violation = F.relu(dist_to_pos - dist_to_neg + margin)

    # For negative labels: want dist_neg < dist_pos - margin
    # Violation: max(0, dist_neg - dist_pos + margin)
    neg_mask = ~pos_mask
    neg_violation = F.relu(dist_to_neg - dist_to_pos + margin)

    # Sum violations where applicable
    loss = (pos_violation * pos_mask.float()).sum() + (neg_violation * neg_mask.float()).sum()

    # Normalize by total number of elements
    return loss / labels.numel()
