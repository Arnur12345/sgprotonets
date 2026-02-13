"""Meta-test evaluation over episodes."""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.iu_xray import IUXRayDataset, IUXRayMultiLabelDataset
from data.episode_sampler import (
    EpisodeSampler,
    BinaryEpisodeSampler,
    unpack_episode,
    unpack_binary_episode,
    binary_episode_collate_fn,
)
from data.class_descriptions import CLASS_DESCRIPTIONS
from models.sgprotonet import SGProtoNet
from evaluation.metrics import (
    confidence_interval,
    confidence_interval_multilabel,
    per_class_accuracy,
    multilabel_metrics,
    multilabel_auc,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def meta_test(
    model: SGProtoNet,
    cfg: DictConfig,
    device: torch.device,
    split: str = "test",
) -> dict[str, float]:
    """Run meta-test evaluation.

    Args:
        model: Trained SGProtoNet.
        cfg: Full configuration.
        device: Target device.
        split: "test" or "val".

    Returns:
        Dict with mean_accuracy, ci_95, and per_class_acc.
    """
    model.eval()
    ep_cfg = cfg.episode

    # Select classes
    if split == "test":
        classes = list(cfg.data.test_classes)
    else:
        classes = list(cfg.data.val_classes)

    dataset = IUXRayDataset(
        data_dir=cfg.data.data_dir,
        split_classes=classes,
        image_size=cfg.data.image_size,
        is_train=False,
    )

    sampler = EpisodeSampler(
        dataset=dataset,
        n_way=ep_cfg.n_way,
        k_shot=ep_cfg.k_shot,
        q_query=ep_cfg.q_query,
        num_episodes=cfg.inference.num_eval_episodes,
        seed=cfg.seed + 2000,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    # Pre-compute class anchors
    descriptions = [CLASS_DESCRIPTIONS.get(c, "") for c in dataset.classes]
    s_cls, _ = model.semantic_encoder(descriptions, device)
    class_anchors = model.semantic_proj(s_cls)

    episode_accuracies: list[float] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        support, query, query_labels = unpack_episode(
            batch, ep_cfg.n_way, ep_cfg.k_shot, ep_cfg.q_query
        )

        support_images = support["image"].to(device)
        query_images = query["image"].to(device)
        support_labels = support["label"].to(device)
        query_labels_dev = query_labels.to(device)

        # Determine text availability
        support_texts = support["report"]
        query_texts = query["report"]
        text_strategy = cfg.inference.text_strategy

        episode_out = model.forward_episode(
            support_images=support_images,
            support_texts=support_texts,
            support_labels=support_labels,
            query_images=query_images,
            query_texts=query_texts,
            n_way=ep_cfg.n_way,
            class_semantic_embeds=class_anchors,
            text_strategy=text_strategy,
        )

        preds = episode_out["logits"].argmax(dim=-1)
        acc = (preds == query_labels_dev).float().mean().item()
        episode_accuracies.append(acc)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(query_labels.numpy())

    mean_acc, ci = confidence_interval(
        episode_accuracies, cfg.inference.confidence_interval
    )

    # Aggregate per-class stats
    all_preds_arr = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels)
    per_class = per_class_accuracy(all_preds_arr, all_labels_arr, ep_cfg.n_way)

    logger.info(f"Meta-test ({split}): {mean_acc:.4f} +/- {ci:.4f}")
    for c, acc in per_class.items():
        logger.info(f"  Class {c}: {acc:.4f}")

    return {
        "mean_accuracy": mean_acc,
        "ci_95": ci,
        "per_class_acc": per_class,
        "episode_accuracies": episode_accuracies,
    }


@torch.no_grad()
def meta_test_multilabel(
    model: SGProtoNet,
    cfg: DictConfig,
    device: torch.device,
    split: str = "test",
) -> dict[str, float]:
    """Run meta-test evaluation for multi-label classification.

    Args:
        model: Trained SGProtoNet with multi-label support.
        cfg: Full configuration.
        device: Target device.
        split: "test" or "val".

    Returns:
        Dict with multi-label metrics: mean_f1, exact_match, micro_auc, etc.
    """
    model.eval()
    ml_cfg = cfg.multilabel

    # Select classes
    if split == "test":
        classes = list(cfg.data.test_classes)
    else:
        classes = list(cfg.data.val_classes)

    # Use multi-label dataset
    dataset = IUXRayMultiLabelDataset(
        data_dir=cfg.data.data_dir,
        split_classes=classes if classes else list(cfg.data.train_classes),
        image_size=cfg.data.image_size,
        is_train=False,
    )

    sampler = BinaryEpisodeSampler(
        dataset=dataset,
        n_labels=ml_cfg.n_labels,
        k_pos=ml_cfg.k_pos,
        k_neg=ml_cfg.k_neg,
        q_query=ml_cfg.q_query,
        num_episodes=cfg.inference.num_eval_episodes,
        min_positives=ml_cfg.get("min_positives", 3),
        min_negatives=ml_cfg.get("min_negatives", 10),
        seed=cfg.seed + 2000,
    )

    # Pre-compute class anchors
    descriptions = [CLASS_DESCRIPTIONS.get(c, "") for c in dataset.classes]
    s_cls, _ = model.semantic_encoder(descriptions, device)
    class_anchors = model.semantic_proj(s_cls)

    # Sample counts for adaptive anchor weighting
    sample_counts = [
        len(dataset.positive_indices[c])
        for c in range(dataset.num_classes)
    ]

    episode_f1s: list[float] = []
    episode_exact_match: list[float] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    n_total = sampler.num_episodes
    for ep_idx, episode_meta in enumerate(sampler):
        indices = episode_meta["indices"]
        samples = [dataset[i] for i in indices]
        batch = binary_episode_collate_fn(samples)

        episode_labels = episode_meta["episode_labels"]
        n_labels = episode_meta["n_labels"]
        k_pos = episode_meta["k_pos"]
        k_neg = episode_meta["k_neg"]
        q_query = episode_meta["q_query"]

        support_pos, support_neg, query, query_labels, _ = unpack_binary_episode(
            batch, episode_labels, n_labels, k_pos, k_neg, q_query
        )

        support_pos_images = support_pos["image"].to(device)
        support_neg_images = support_neg["image"].to(device)
        query_images = query["image"].to(device)
        query_labels_dev = query_labels.to(device)

        episode_anchors = class_anchors[episode_labels]
        episode_sample_counts = [sample_counts[l] for l in episode_labels]

        episode_out = model.forward_binary_episode(
            support_pos_images=support_pos_images,
            support_pos_texts=support_pos["report"],
            support_neg_images=support_neg_images,
            support_neg_texts=support_neg["report"],
            query_images=query_images,
            query_texts=query["report"],
            n_labels=n_labels,
            k_pos=k_pos,
            k_neg=k_neg,
            class_semantic_embeds=episode_anchors,
            label_sample_counts=episode_sample_counts,
        )

        logits = episode_out["binary_logits"]
        probs = torch.sigmoid(logits)
        preds = (probs > ml_cfg.get("threshold", 0.5)).float()

        # Compute per-episode metrics
        tp = (preds * query_labels_dev).sum(dim=1)
        fp = (preds * (1 - query_labels_dev)).sum(dim=1)
        fn = ((1 - preds) * query_labels_dev).sum(dim=1)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mean_f1 = torch.nan_to_num(f1, nan=0.0).mean().item()

        exact = (preds == query_labels_dev).all(dim=1).float().mean().item()

        episode_f1s.append(mean_f1)
        episode_exact_match.append(exact)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(query_labels.numpy())

        # Free GPU memory periodically
        if (ep_idx + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"  Episode {ep_idx + 1}/{n_total} — running F1: {np.mean(episode_f1s):.4f}")

    # Compute confidence intervals
    mean_f1, ci_f1 = confidence_interval_multilabel(
        episode_f1s, cfg.inference.confidence_interval
    )
    mean_exact, ci_exact = confidence_interval_multilabel(
        episode_exact_match, cfg.inference.confidence_interval
    )

    # Aggregate all predictions for global metrics
    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_preds_arr = np.concatenate(all_preds, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)

    global_metrics = multilabel_metrics(all_preds_arr, all_labels_arr)
    auc_metrics = multilabel_auc(all_probs_arr, all_labels_arr)

    logger.info(f"Meta-test Multi-Label ({split}):")
    logger.info(f"  Mean F1: {mean_f1:.4f} +/- {ci_f1:.4f}")
    logger.info(f"  Exact Match: {mean_exact:.4f} +/- {ci_exact:.4f}")
    logger.info(f"  Micro F1: {global_metrics['micro_f1']:.4f}")
    logger.info(f"  Macro F1: {global_metrics['macro_f1']:.4f}")
    logger.info(f"  Micro AUC: {auc_metrics['micro_auc']:.4f}")
    logger.info(f"  Macro AUC: {auc_metrics['macro_auc']:.4f}")

    return {
        "mean_f1": mean_f1,
        "ci_f1": ci_f1,
        "mean_exact_match": mean_exact,
        "ci_exact_match": ci_exact,
        "micro_f1": global_metrics["micro_f1"],
        "macro_f1": global_metrics["macro_f1"],
        "micro_precision": global_metrics["micro_precision"],
        "micro_recall": global_metrics["micro_recall"],
        "micro_auc": auc_metrics["micro_auc"],
        "macro_auc": auc_metrics["macro_auc"],
        "per_label_metrics": global_metrics["per_label_metrics"],
        "per_label_auc": auc_metrics["per_label_auc"],
        "episode_f1s": episode_f1s,
        "episode_exact_match": episode_exact_match,
    }
