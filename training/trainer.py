"""Main training loop (two-phase)."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
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
from training.losses import alignment_loss, vis2sem_loss
from training.episode_loop import episode_step, binary_episode_step
from training.schedulers import build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Two-phase trainer for SGProtoNet.

    Phase 1: Modality alignment (non-episodic, InfoNCE contrastive).
    Phase 2: Episodic meta-training (prototypical + alignment + consistency).

    Args:
        cfg: Full configuration (DictConfig).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = SGProtoNet(cfg).to(self.device)

        # Mixed precision training
        self.use_amp = cfg.training.get("use_amp", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Setup logging
        self.wandb_run = None
        if cfg.logging.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.training.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_metric = 0.0

    def _build_optimizer(self, phase: int) -> optim.Optimizer:
        """Build optimizer for trainable parameters only."""
        if phase == 1:
            lr = self.cfg.training.phase1.lr
            wd = self.cfg.training.phase1.weight_decay
        else:
            lr = self.cfg.training.phase2.lr
            wd = self.cfg.training.phase2.weight_decay

        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=lr, weight_decay=wd)

    def _get_class_anchors(self, classes: list[str]) -> torch.Tensor:
        """Encode class-level text descriptions as semantic anchors.

        Args:
            classes: List of class names.

        Returns:
            Tensor of shape (n_classes, d_model).
        """
        descriptions = [CLASS_DESCRIPTIONS.get(c, "") for c in classes]
        with torch.no_grad():
            s_cls, _ = self.model.semantic_encoder(descriptions, self.device)
            anchors = self.model.semantic_proj(s_cls)
        return anchors

    def train_phase1(self) -> None:
        """Phase 1: Modality alignment training (non-episodic)."""
        if not self.cfg.training.phase1.enabled:
            logger.info("Phase 1 disabled, skipping.")
            return

        logger.info("=== Phase 1: Modality Alignment ===")
        p1_cfg = self.cfg.training.phase1

        # Build dataset and dataloader
        dataset = IUXRayDataset(
            data_dir=self.cfg.data.data_dir,
            split_classes=list(self.cfg.data.train_classes),
            image_size=self.cfg.data.image_size,
            is_train=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=p1_cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = self._build_optimizer(phase=1)
        scheduler = build_scheduler(optimizer, self.cfg.training.scheduler, p1_cfg.num_epochs)

        self.model.train()
        for epoch in range(p1_cfg.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                images = batch["image"].to(self.device)
                texts = batch["report"]

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.model(images, texts)

                    # InfoNCE alignment loss
                    loss = alignment_loss(out["v_cls_proj"], out["s_proj"])

                    # Vis2Sem auxiliary loss
                    predicted_s = self.model.vis2sem(out["v_cls_proj"])
                    loss = loss + 0.5 * vis2sem_loss(predicted_s, out["s_proj"])

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Phase 1 Epoch {epoch+1}/{p1_cfg.num_epochs} — Loss: {avg_loss:.4f}")

            if self.wandb_run:
                self.wandb_run.log({"phase1/loss": avg_loss, "phase1/epoch": epoch + 1})

    def train_phase2(self) -> None:
        """Phase 2: Episodic meta-training."""
        logger.info("=== Phase 2: Episodic Meta-Training ===")
        p2_cfg = self.cfg.training.phase2
        ep_cfg = self.cfg.episode

        # Build training dataset and sampler
        train_dataset = IUXRayDataset(
            data_dir=self.cfg.data.data_dir,
            split_classes=list(self.cfg.data.train_classes),
            image_size=self.cfg.data.image_size,
            is_train=True,
        )
        train_sampler = EpisodeSampler(
            dataset=train_dataset,
            n_way=ep_cfg.n_way,
            k_shot=ep_cfg.k_shot,
            q_query=ep_cfg.q_query,
            num_episodes=p2_cfg.episodes_per_epoch,
            seed=self.cfg.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        # Validation dataset
        val_dataset = IUXRayDataset(
            data_dir=self.cfg.data.data_dir,
            split_classes=list(self.cfg.data.val_classes),
            image_size=self.cfg.data.image_size,
            is_train=False,
        )

        optimizer = self._build_optimizer(phase=2)
        scheduler = build_scheduler(optimizer, self.cfg.training.scheduler, p2_cfg.num_epochs)

        # Pre-compute class anchors for train classes
        class_anchors = self._get_class_anchors(train_dataset.classes)

        for epoch in range(p2_cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_episodes = 0

            for batch in train_loader:
                support, query, query_labels = unpack_episode(
                    batch, ep_cfg.n_way, ep_cfg.k_shot, ep_cfg.q_query
                )

                support_images = support["image"].to(self.device)
                query_images = query["image"].to(self.device)
                support_labels = support["label"].to(self.device)
                query_labels = query_labels.to(self.device)

                # Use per-image text when available
                support_texts = support["report"]
                query_texts = query["report"]

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    result = episode_step(
                        model=self.model,
                        support_images=support_images,
                        support_texts=support_texts,
                        support_labels=support_labels,
                        query_images=query_images,
                        query_texts=query_texts,
                        query_labels=query_labels,
                        n_way=ep_cfg.n_way,
                        lambda_align=p2_cfg.lambda_align,
                        lambda_consist=p2_cfg.lambda_consist,
                        class_semantic_embeds=class_anchors,
                    )

                self.scaler.scale(result["loss"]).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += result["loss"].item()
                epoch_acc += result["accuracy"].item()
                n_episodes += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_episodes, 1)
            avg_acc = epoch_acc / max(n_episodes, 1)
            logger.info(
                f"Phase 2 Epoch {epoch+1}/{p2_cfg.num_epochs} — "
                f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
            )

            if self.wandb_run:
                self.wandb_run.log({
                    "phase2/loss": avg_loss,
                    "phase2/accuracy": avg_acc,
                    "phase2/epoch": epoch + 1,
                })

            # Validation
            if (epoch + 1) % self.cfg.training.val_every_n_epochs == 0:
                val_acc = self._validate(val_dataset)
                logger.info(f"  Validation Acc: {val_acc:.4f}")
                if self.wandb_run:
                    self.wandb_run.log({"val/accuracy": val_acc})

                if val_acc > self.best_val_metric:
                    self.best_val_metric = val_acc
                    if self.cfg.training.save_best:
                        self._save_checkpoint("best.pt", epoch, val_acc)

            # Periodic save
            if (epoch + 1) % self.cfg.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pt", epoch, avg_acc)

    @torch.no_grad()
    def _validate(self, val_dataset: IUXRayDataset) -> float:
        """Run validation episodes.

        Args:
            val_dataset: Validation dataset.

        Returns:
            Mean accuracy over validation episodes.
        """
        self.model.eval()
        ep_cfg = self.cfg.episode

        # Use min of n_way or number of available classes
        val_n_way = min(ep_cfg.n_way, len(val_dataset.classes))

        val_sampler = EpisodeSampler(
            dataset=val_dataset,
            n_way=val_n_way,
            k_shot=ep_cfg.k_shot,
            q_query=ep_cfg.q_query,
            num_episodes=self.cfg.training.val_episodes,
            seed=self.cfg.seed + 1000,  # Different seed from training
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
        )

        class_anchors = self._get_class_anchors(val_dataset.classes)

        total_acc = 0.0
        n_episodes = 0

        for batch in val_loader:
            support, query, query_labels = unpack_episode(
                batch, val_n_way, ep_cfg.k_shot, ep_cfg.q_query
            )

            support_images = support["image"].to(self.device)
            query_images = query["image"].to(self.device)
            support_labels = support["label"].to(self.device)
            query_labels = query_labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                episode_out = self.model.forward_episode(
                    support_images=support_images,
                    support_texts=support["report"],
                    support_labels=support_labels,
                    query_images=query_images,
                    query_texts=query["report"],
                    n_way=val_n_way,
                    class_semantic_embeds=class_anchors,
                )

            preds = episode_out["logits"].argmax(dim=-1)
            acc = (preds == query_labels).float().mean().item()
            total_acc += acc
            n_episodes += 1

        return total_acc / max(n_episodes, 1)

    def train_phase2_multilabel(self) -> None:
        """Phase 2: Episodic meta-training with binary decomposition for multi-label."""
        logger.info("=== Phase 2: Multi-Label Episodic Meta-Training ===")
        p2_cfg = self.cfg.training.phase2
        ml_cfg = self.cfg.multilabel

        # Build multi-label training dataset
        train_dataset = IUXRayMultiLabelDataset(
            data_dir=self.cfg.data.data_dir,
            split_classes=list(self.cfg.data.train_classes),
            image_size=self.cfg.data.image_size,
            is_train=True,
        )

        # Build multi-label validation dataset (on val_classes, NOT train_classes)
        val_classes = list(self.cfg.data.val_classes)
        if val_classes:
            val_dataset = IUXRayMultiLabelDataset(
                data_dir=self.cfg.data.data_dir,
                split_classes=val_classes,
                image_size=self.cfg.data.image_size,
                is_train=False,
            )
            logger.info(f"Validation dataset: {len(val_dataset)} samples, "
                        f"{val_dataset.num_classes} classes: {val_dataset.classes}")
        else:
            val_dataset = None
            logger.warning("No val_classes defined — validation will be skipped.")

        # Log class sample counts
        sample_counts = train_dataset.get_class_sample_counts()
        logger.info("Training set class sample counts:")
        for cls_name, counts in sample_counts.items():
            logger.info(f"  {cls_name}: pos={counts['positive']}, neg={counts['negative']}")

        # Binary episode sampler
        train_sampler = BinaryEpisodeSampler(
            dataset=train_dataset,
            n_labels=ml_cfg.n_labels,
            k_pos=ml_cfg.k_pos,
            k_neg=ml_cfg.k_neg,
            q_query=ml_cfg.q_query,
            num_episodes=p2_cfg.episodes_per_epoch,
            min_positives=ml_cfg.get("min_positives", 3),
            min_negatives=ml_cfg.get("min_negatives", 10),
            seed=self.cfg.seed,
        )

        # Custom collate for binary episodes
        def collate_with_metadata(indices_and_meta):
            """Collate samples and preserve episode metadata."""
            # indices_and_meta is a dict from BinaryEpisodeSampler
            # We need to fetch the actual samples
            indices = indices_and_meta["indices"]
            samples = [train_dataset[i] for i in indices]
            batch = binary_episode_collate_fn(samples)
            batch["_episode_labels"] = indices_and_meta["episode_labels"]
            batch["_n_labels"] = indices_and_meta["n_labels"]
            batch["_k_pos"] = indices_and_meta["k_pos"]
            batch["_k_neg"] = indices_and_meta["k_neg"]
            batch["_q_query"] = indices_and_meta["q_query"]
            return batch

        # Note: BinaryEpisodeSampler yields dicts, so we iterate directly
        optimizer = self._build_optimizer(phase=2)
        scheduler = build_scheduler(optimizer, self.cfg.training.scheduler, p2_cfg.num_epochs)

        # Pre-compute class anchors for ALL classes
        all_class_anchors = self._get_class_anchors(train_dataset.classes)

        # Get sample counts for adaptive anchor weighting
        all_sample_counts = [
            len(train_dataset.positive_indices[c])
            for c in range(train_dataset.num_classes)
        ]

        for epoch in range(p2_cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_f1 = 0.0
            epoch_exact_match = 0.0
            n_episodes = 0

            # Iterate through sampler directly
            for episode_meta in train_sampler:
                # Fetch samples for this episode
                indices = episode_meta["indices"]
                samples = [train_dataset[i] for i in indices]
                batch = binary_episode_collate_fn(samples)

                episode_labels = episode_meta["episode_labels"]
                n_labels = episode_meta["n_labels"]
                k_pos = episode_meta["k_pos"]
                k_neg = episode_meta["k_neg"]
                q_query = episode_meta["q_query"]

                # Unpack episode
                support_pos, support_neg, query, query_labels, _ = unpack_binary_episode(
                    batch, episode_labels, n_labels, k_pos, k_neg, q_query
                )

                # Move to device
                support_pos_images = support_pos["image"].to(self.device)
                support_neg_images = support_neg["image"].to(self.device)
                query_images = query["image"].to(self.device)
                query_labels = query_labels.to(self.device)

                # Get class anchors for episode labels
                episode_anchors = all_class_anchors[episode_labels]

                # Get sample counts for episode labels
                episode_sample_counts = [all_sample_counts[l] for l in episode_labels]

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    result = binary_episode_step(
                        model=self.model,
                        support_pos_images=support_pos_images,
                        support_pos_texts=support_pos["report"],
                        support_neg_images=support_neg_images,
                        support_neg_texts=support_neg["report"],
                        query_images=query_images,
                        query_texts=query["report"],
                        query_labels=query_labels,
                        n_labels=n_labels,
                        k_pos=k_pos,
                        k_neg=k_neg,
                        lambda_align=p2_cfg.lambda_align,
                        lambda_consist=p2_cfg.lambda_consist,
                        lambda_margin=ml_cfg.get("lambda_margin", 0.1),
                        class_semantic_embeds=episode_anchors,
                        use_focal_loss=ml_cfg.get("use_focal_loss", True),
                        focal_gamma=ml_cfg.get("focal_gamma", 2.0),
                        focal_alpha=ml_cfg.get("focal_alpha", 0.25),
                        label_sample_counts=episode_sample_counts,
                    )

                self.scaler.scale(result["loss"]).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += result["loss"].item()
                epoch_f1 += result["mean_f1"].item()
                epoch_exact_match += result["exact_match_ratio"].item()
                n_episodes += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_episodes, 1)
            avg_f1 = epoch_f1 / max(n_episodes, 1)
            avg_exact = epoch_exact_match / max(n_episodes, 1)

            logger.info(
                f"Phase 2 Epoch {epoch+1}/{p2_cfg.num_epochs} — "
                f"Loss: {avg_loss:.4f}, F1: {avg_f1:.4f}, ExactMatch: {avg_exact:.4f}"
            )

            if self.wandb_run:
                self.wandb_run.log({
                    "phase2/loss": avg_loss,
                    "phase2/mean_f1": avg_f1,
                    "phase2/exact_match": avg_exact,
                    "phase2/epoch": epoch + 1,
                })

            # Validation (on val_classes, not train_classes)
            if (epoch + 1) % self.cfg.training.val_every_n_epochs == 0 and val_dataset is not None:
                val_metrics = self._validate_multilabel(val_dataset)
                logger.info(
                    f"  Validation — F1: {val_metrics['mean_f1']:.4f}, "
                    f"ExactMatch: {val_metrics['exact_match']:.4f}"
                )
                if self.wandb_run:
                    self.wandb_run.log({
                        "val/mean_f1": val_metrics["mean_f1"],
                        "val/exact_match": val_metrics["exact_match"],
                    })

                if val_metrics["mean_f1"] > self.best_val_metric:
                    self.best_val_metric = val_metrics["mean_f1"]
                    if self.cfg.training.save_best:
                        self._save_checkpoint("best.pt", epoch, val_metrics["mean_f1"])

            # Periodic save
            if (epoch + 1) % self.cfg.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pt", epoch, avg_f1)

    @torch.no_grad()
    def _validate_multilabel(self, val_dataset: IUXRayMultiLabelDataset) -> dict:
        """Run validation episodes for multi-label.

        Args:
            val_dataset: Validation dataset (must use val_classes, not train_classes).

        Returns:
            Dict with mean_f1 and exact_match metrics.
        """
        self.model.eval()
        ml_cfg = self.cfg.multilabel

        # Adapt n_labels to available valid labels in val set
        val_n_labels = min(ml_cfg.n_labels, val_dataset.num_classes)

        # Create validation sampler with different seed
        val_sampler = BinaryEpisodeSampler(
            dataset=val_dataset,
            n_labels=val_n_labels,
            k_pos=ml_cfg.k_pos,
            k_neg=ml_cfg.k_neg,
            q_query=ml_cfg.q_query,
            num_episodes=self.cfg.training.val_episodes,
            min_positives=ml_cfg.get("min_positives", 3),
            min_negatives=ml_cfg.get("min_negatives", 10),
            seed=self.cfg.seed + 1000,
        )

        all_class_anchors = self._get_class_anchors(val_dataset.classes)
        all_sample_counts = [
            len(val_dataset.positive_indices[c])
            for c in range(val_dataset.num_classes)
        ]

        total_f1 = 0.0
        total_exact = 0.0
        n_episodes = 0

        for episode_meta in val_sampler:
            indices = episode_meta["indices"]
            samples = [val_dataset[i] for i in indices]
            batch = binary_episode_collate_fn(samples)

            episode_labels = episode_meta["episode_labels"]
            n_labels = episode_meta["n_labels"]
            k_pos = episode_meta["k_pos"]
            k_neg = episode_meta["k_neg"]
            q_query = episode_meta["q_query"]

            support_pos, support_neg, query, query_labels, _ = unpack_binary_episode(
                batch, episode_labels, n_labels, k_pos, k_neg, q_query
            )

            support_pos_images = support_pos["image"].to(self.device)
            support_neg_images = support_neg["image"].to(self.device)
            query_images = query["image"].to(self.device)
            query_labels = query_labels.to(self.device)

            episode_anchors = all_class_anchors[episode_labels]
            episode_sample_counts = [all_sample_counts[l] for l in episode_labels]

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                episode_out = self.model.forward_binary_episode(
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

            # Compute metrics
            tp = (preds * query_labels).sum(dim=1)
            fp = (preds * (1 - query_labels)).sum(dim=1)
            fn = ((1 - preds) * query_labels).sum(dim=1)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            mean_f1 = torch.nan_to_num(f1, nan=0.0).mean().item()

            exact_match = (preds == query_labels).all(dim=1).float().mean().item()

            total_f1 += mean_f1
            total_exact += exact_match
            n_episodes += 1

        return {
            "mean_f1": total_f1 / max(n_episodes, 1),
            "exact_match": total_exact / max(n_episodes, 1),
        }

    def _save_checkpoint(self, filename: str, epoch: int, accuracy: float) -> None:
        """Save model checkpoint."""
        path = self.ckpt_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "accuracy": accuracy,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def train(self) -> None:
        """Run full two-phase training."""
        self.train_phase1()

        # Check if multi-label mode is enabled
        if self.cfg.get("multilabel", {}).get("enabled", False):
            self.train_phase2_multilabel()
        else:
            self.train_phase2()

        if self.wandb_run:
            self.wandb_run.finish()

        logger.info(f"Training complete. Best val metric: {self.best_val_metric:.4f}")
