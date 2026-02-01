"""Meta-test evaluation over episodes."""

import logging

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.iu_xray import IUXRayDataset
from data.episode_sampler import EpisodeSampler, unpack_episode
from data.class_descriptions import CLASS_DESCRIPTIONS
from models.sgprotonet import SGProtoNet
from evaluation.metrics import confidence_interval, per_class_accuracy

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
        num_workers=4,
        pin_memory=True,
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
