"""Single episode forward/backward logic.

IMPORTANT — Privileged-text formulation (Framing A):
  Per-image radiology reports are a known source of label leakage on IU-CXR
  because labels were extracted from those reports by CheXbert. To prevent the
  model from short-circuiting via text, reports never enter `forward_episode`
  or the prototype computation. They are used ONLY as auxiliary supervision
  (alignment + vis2sem losses) on the support set, which trains the visual
  stream to produce report-aligned features without seeing them at inference.
"""

from __future__ import annotations

import torch

from models.sgprotonet import SGProtoNet
from training.losses import (
    prototypical_loss,
    alignment_loss,
    vis2sem_loss,
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
    lambda_vis2sem: float = 0.5,
    class_semantic_embeds: torch.Tensor | None = None,
    text_strategy: str = "class_anchors",
) -> dict[str, torch.Tensor]:
    """Run a single episode forward pass and compute losses.

    Reports (`support_texts`) — if provided — are used ONLY as auxiliary
    supervision via alignment + vis2sem losses. They are NEVER passed into
    `forward_episode`, so they never reach the prototypes or the query logits.
    This keeps train- and test-time inference identical: image + class anchors.

    Args:
        model: SGProtoNet model.
        support_images: (n_way * k_shot, 3, H, W).
        support_texts: Optional support reports. Used only as privileged
            supervision; does NOT reach `forward_episode`.
        support_labels: (n_way * k_shot,).
        query_images: (n_way * q_query, 3, H, W).
        query_texts: Ignored at the prototype path. Accepted for API
            compatibility but never used.
        query_labels: (n_way * q_query,).
        n_way: Number of classes.
        lambda_align: Weight for alignment (InfoNCE) loss.
        lambda_consist: Weight for consistency loss (kept for compatibility,
            but disabled here because there is no per-image text in `f_final`).
        lambda_vis2sem: Weight for vis2sem auxiliary loss.
        class_semantic_embeds: Class anchors (n_way, d_model). Required.
        text_strategy: Text strategy for inference. Forced to 'class_anchors'.

    Returns:
        Dict with loss, l_proto, l_align, l_consist, l_vis2sem, accuracy.
    """
    # Prototype path: image + class anchors ONLY. Reports never enter here.
    episode_out = model.forward_episode(
        support_images=support_images,
        support_texts=None,
        support_labels=support_labels,
        query_images=query_images,
        query_texts=None,
        n_way=n_way,
        class_semantic_embeds=class_semantic_embeds,
        text_strategy="class_anchors",
    )

    logits = episode_out["logits"]
    support_out = episode_out["support_out"]

    # Primary loss: prototypical cross-entropy
    l_proto = prototypical_loss(logits, query_labels)

    # Auxiliary losses use reports as PRIVILEGED supervision on the support
    # visual features. Reports are encoded via the semantic encoder/projection
    # but the resulting embedding is only used as a target for v_cls_proj and
    # vis2sem. It does NOT propagate into f_final or the prototypes.
    device = logits.device
    l_align = torch.zeros((), device=device)
    l_vis2sem = torch.zeros((), device=device)

    if support_texts is not None and any(t.strip() for t in support_texts):
        with torch.no_grad():
            # Detach the text branch from the visual computation graph entirely.
            s_cls, _ = model.semantic_encoder(support_texts, device)
        s_proj_target = model.semantic_proj(s_cls)

        # InfoNCE between visual CLS (from the prototype path) and report embeddings.
        l_align = alignment_loss(support_out["v_cls_proj"], s_proj_target)

        # Train vis2sem to predict the report embedding from the visual features.
        # This is a bonus head that NEVER affects f_final at inference under
        # text_strategy='class_anchors'; it just regularizes v_cls_proj.
        predicted_s = model.vis2sem(support_out["v_cls_proj"])
        l_vis2sem = vis2sem_loss(predicted_s, s_proj_target.detach())

    # Consistency loss is meaningless in this formulation because f_final is
    # already built from class anchors (not per-image text), so f_final and
    # v_cls_proj are deliberately *meant* to differ. Kept at 0 for clarity.
    l_consist = torch.zeros((), device=device)

    # Total loss
    loss = (
        l_proto
        + lambda_align * l_align
        + lambda_vis2sem * l_vis2sem
        + lambda_consist * l_consist
    )

    # Accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean()

    return {
        "loss": loss,
        "l_proto": l_proto,
        "l_align": l_align,
        "l_consist": l_consist,
        "l_vis2sem": l_vis2sem,
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
    lambda_vis2sem: float = 0.5,
    class_semantic_embeds: torch.Tensor | None = None,
    text_strategy: str = "class_anchors",
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_sample_counts: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Binary decomposition episode (multi-label) — privileged-text version.

    Same principle as `episode_step`: per-image reports are used only for the
    alignment + vis2sem auxiliary losses on the *positive* support set. They
    do NOT reach `forward_binary_episode` or the prototypes.
    """
    # Prototype path: images + class anchors only.
    episode_out = model.forward_binary_episode(
        support_pos_images=support_pos_images,
        support_pos_texts=None,
        support_neg_images=support_neg_images,
        support_neg_texts=None,
        query_images=query_images,
        query_texts=None,
        n_labels=n_labels,
        k_pos=k_pos,
        k_neg=k_neg,
        class_semantic_embeds=class_semantic_embeds,
        text_strategy="class_anchors",
        label_sample_counts=label_sample_counts,
    )

    logits = episode_out["binary_logits"]
    support_pos_out = episode_out["support_pos_out"]
    query_out = episode_out["query_out"]
    prototypes_pos = episode_out["prototypes_pos"]
    prototypes_neg = episode_out["prototypes_neg"]

    # Primary loss
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

    # Auxiliary losses on positive support reports (privileged supervision).
    device = logits.device
    l_align = torch.zeros((), device=device)
    l_vis2sem = torch.zeros((), device=device)

    if support_pos_texts is not None and any(t.strip() for t in support_pos_texts):
        with torch.no_grad():
            s_cls, _ = model.semantic_encoder(support_pos_texts, device)
        s_proj_target = model.semantic_proj(s_cls)

        l_align = alignment_loss(support_pos_out["v_cls_proj"], s_proj_target)
        predicted_s = model.vis2sem(support_pos_out["v_cls_proj"])
        l_vis2sem = vis2sem_loss(predicted_s, s_proj_target.detach())

    # Consistency loss disabled — see episode_step explanation.
    l_consist = torch.zeros((), device=device)

    loss = (
        l_binary
        + lambda_align * l_align
        + lambda_vis2sem * l_vis2sem
        + lambda_consist * l_consist
        + lambda_margin * l_margin
    )

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        per_label_correct = (preds == query_labels).float()
        per_label_accuracy = per_label_correct.mean(dim=0)
        mean_label_accuracy = per_label_accuracy.mean()
        exact_match = per_label_correct.all(dim=1).float().mean()

        tp = (preds * query_labels).sum(dim=1)
        fp = (preds * (1 - query_labels)).sum(dim=1)
        fn = ((1 - preds) * query_labels).sum(dim=1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mean_f1 = torch.nan_to_num(f1.mean(), nan=0.0)

    return {
        "loss": loss,
        "l_binary": l_binary,
        "l_margin": l_margin,
        "l_align": l_align,
        "l_consist": l_consist,
        "l_vis2sem": l_vis2sem,
        "per_label_accuracy": per_label_accuracy,
        "mean_label_accuracy": mean_label_accuracy,
        "exact_match_ratio": exact_match,
        "mean_f1": mean_f1,
    }
