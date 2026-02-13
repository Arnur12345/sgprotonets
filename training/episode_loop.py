"""Single episode forward/backward logic."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.sgprotonet import SGProtoNet
from training.losses import (
    prototypical_loss,
    alignment_loss,
    consistency_loss,
    multilabel_binary_loss,
    focal_multilabel_loss,
    prototype_margin_loss,
)


def episode_step(
    model: SGProtoNet,
    support_images: torch.Tensor,
    support_texts: list[str] | None,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_texts: list[str] | None,
    query_labels: torch.Tensor,
    n_way: int,
    lambda_align: float = 0.5,
    lambda_consist: float = 0.1,
    class_semantic_embeds: torch.Tensor | None = None,
    text_strategy: str = "class_anchors",
) -> dict[str, torch.Tensor]:
    """Run a single episode forward pass and compute losses.

    Args:
        model: SGProtoNet model.
        support_images: (n_way * k_shot, 3, H, W).
        support_texts: Support texts or None.
        support_labels: (n_way * k_shot,).
        query_images: (n_way * q_query, 3, H, W).
        query_texts: Query texts or None.
        query_labels: (n_way * q_query,).
        n_way: Number of classes.
        lambda_align: Weight for alignment loss.
        lambda_consist: Weight for consistency loss.
        class_semantic_embeds: Optional class anchors (n_way, d_model).
        text_strategy: Text strategy for inference.

    Returns:
        Dict with loss, l_proto, l_align, l_consist, accuracy.
    """
    episode_out = model.forward_episode(
        support_images=support_images,
        support_texts=support_texts,
        support_labels=support_labels,
        query_images=query_images,
        query_texts=query_texts,
        n_way=n_way,
        class_semantic_embeds=class_semantic_embeds,
        text_strategy=text_strategy,
    )

    logits = episode_out["logits"]
    support_out = episode_out["support_out"]
    query_out = episode_out["query_out"]

    # Primary loss: prototypical cross-entropy
    l_proto = prototypical_loss(logits, query_labels)

    # Alignment loss on support set (if text available)
    l_align = torch.tensor(0.0, device=logits.device)
    if support_texts is not None and any(t.strip() for t in support_texts):
        l_align = alignment_loss(support_out["v_cls_proj"], support_out["s_proj"])

    # Consistency loss on support set
    l_consist = consistency_loss(support_out["f_final"], support_out["v_cls_proj"])

    # Total loss
    loss = l_proto + lambda_align * l_align + lambda_consist * l_consist

    # Accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean()

    return {
        "loss": loss,
        "l_proto": l_proto,
        "l_align": l_align,
        "l_consist": l_consist,
        "accuracy": accuracy,
    }


def binary_episode_step(
    model: SGProtoNet,
    support_pos_images: torch.Tensor,
    support_pos_texts: list[str] | None,
    support_neg_images: torch.Tensor,
    support_neg_texts: list[str] | None,
    query_images: torch.Tensor,
    query_texts: list[str] | None,
    query_labels: torch.Tensor,
    n_labels: int,
    k_pos: int,
    k_neg: int,
    lambda_align: float = 0.5,
    lambda_consist: float = 0.1,
    lambda_margin: float = 0.1,
    class_semantic_embeds: torch.Tensor | None = None,
    text_strategy: str = "class_anchors",
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_sample_counts: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Run a binary decomposition episode forward pass and compute losses.

    Args:
        model: SGProtoNet model.
        support_pos_images: (n_labels * k_pos, 3, H, W) positive support.
        support_pos_texts: Positive support texts or None.
        support_neg_images: (n_labels * k_neg, 3, H, W) negative support.
        support_neg_texts: Negative support texts or None.
        query_images: (n_query, 3, H, W) query images.
        query_texts: Query texts or None.
        query_labels: (n_query, n_labels) multi-hot ground truth.
        n_labels: Number of labels in episode.
        k_pos: Positive support examples per label.
        k_neg: Negative support examples per label.
        lambda_align: Weight for alignment loss.
        lambda_consist: Weight for consistency loss.
        lambda_margin: Weight for prototype margin loss.
        class_semantic_embeds: Optional class anchors (n_labels, d_model).
        text_strategy: Text strategy for inference.
        use_focal_loss: If True, use focal loss; otherwise standard BCE.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
        label_sample_counts: Total positive samples per label (for adaptive anchor).

    Returns:
        Dict with loss components and metrics.
    """
    episode_out = model.forward_binary_episode(
        support_pos_images=support_pos_images,
        support_pos_texts=support_pos_texts,
        support_neg_images=support_neg_images,
        support_neg_texts=support_neg_texts,
        query_images=query_images,
        query_texts=query_texts,
        n_labels=n_labels,
        k_pos=k_pos,
        k_neg=k_neg,
        class_semantic_embeds=class_semantic_embeds,
        text_strategy=text_strategy,
        label_sample_counts=label_sample_counts,
    )

    logits = episode_out["binary_logits"]
    support_pos_out = episode_out["support_pos_out"]
    support_neg_out = episode_out["support_neg_out"]
    query_out = episode_out["query_out"]
    prototypes_pos = episode_out["prototypes_pos"]
    prototypes_neg = episode_out["prototypes_neg"]

    # Primary loss: multi-label binary classification
    if use_focal_loss:
        l_binary = focal_multilabel_loss(
            logits, query_labels, gamma=focal_gamma, alpha=focal_alpha
        )
    else:
        l_binary = multilabel_binary_loss(logits, query_labels)

    # Margin loss for prototype separation
    l_margin = prototype_margin_loss(
        query_out["f_final"],
        prototypes_pos,
        prototypes_neg,
        query_labels,
    )

    # Alignment loss on positive support (if text available)
    l_align = torch.tensor(0.0, device=logits.device)
    if support_pos_texts is not None and any(t.strip() for t in support_pos_texts):
        l_align = alignment_loss(
            support_pos_out["v_cls_proj"],
            support_pos_out["s_proj"],
        )

    # Consistency loss on positive support
    l_consist = consistency_loss(
        support_pos_out["f_final"],
        support_pos_out["v_cls_proj"],
    )

    # Total loss
    loss = (
        l_binary
        + lambda_align * l_align
        + lambda_consist * l_consist
        + lambda_margin * l_margin
    )

    # Compute metrics
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Per-label accuracy
        per_label_correct = (preds == query_labels).float()
        per_label_accuracy = per_label_correct.mean(dim=0)  # (n_labels,)
        mean_label_accuracy = per_label_accuracy.mean()

        # Exact match ratio (all labels correct for a sample)
        exact_match = per_label_correct.all(dim=1).float().mean()

        # Sample-wise F1
        tp = (preds * query_labels).sum(dim=1)
        fp = (preds * (1 - query_labels)).sum(dim=1)
        fn = ((1 - preds) * query_labels).sum(dim=1)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mean_f1 = f1.mean()

        # Handle edge case where sample has no positive labels
        # (F1 is undefined, but we computed 0/0 = nan)
        mean_f1 = torch.nan_to_num(mean_f1, nan=0.0)

    return {
        "loss": loss,
        "l_binary": l_binary,
        "l_margin": l_margin,
        "l_align": l_align,
        "l_consist": l_consist,
        "per_label_accuracy": per_label_accuracy,
        "mean_label_accuracy": mean_label_accuracy,
        "exact_match_ratio": exact_match,
        "mean_f1": mean_f1,
    }
