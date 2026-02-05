"""Single episode forward/backward logic."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.sgprotonet import SGProtoNet
from training.losses import prototypical_loss, alignment_loss, consistency_loss


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
