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
