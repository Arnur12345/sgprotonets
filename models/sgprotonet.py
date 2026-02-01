"""Full SGProtoNet model — top-level nn.Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.visual_encoder import VisualEncoder
from models.semantic_encoder import SemanticEncoder
from models.projections import VisualProjection, SemanticProjection
from models.sgam import SGAM
from models.fusion import GatedFusion
from models.prototypes import PrototypeComputation, compute_distances
from models.vis2sem import Vis2Sem


class SGProtoNet(nn.Module):
    """Semantic Guided Prototypical Network.

    End-to-end model that encodes images and text, fuses them via SGAM
    and gated fusion, computes prototypes, and classifies queries.

    Args:
        cfg: Full model config (DictConfig).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        model_cfg = cfg.model

        # Encoders
        self.visual_encoder = VisualEncoder(model_cfg)
        self.semantic_encoder = SemanticEncoder(model_cfg)

        # Projections
        self.visual_proj = VisualProjection(model_cfg)
        self.semantic_proj = SemanticProjection(model_cfg)

        # SGAM
        self.sgam = SGAM(
            d_model=model_cfg.d_model,
            num_heads=model_cfg.sgam.num_heads,
            dropout=model_cfg.sgam.dropout,
        )

        # Gated Fusion
        self.fusion = GatedFusion(
            d_model=model_cfg.d_model,
            dropout=model_cfg.fusion.dropout,
        )

        # Prototype computation
        self.prototype = PrototypeComputation(mode=model_cfg.prototype_mode)

        # Vis2Sem for text-free inference
        self.vis2sem = Vis2Sem(
            d_model=model_cfg.d_model,
            d_hidden=model_cfg.vis2sem.d_hidden,
        )

        # Config
        self.distance = model_cfg.distance
        self.d_model = model_cfg.d_model

    def encode_multimodal(
        self,
        images: torch.Tensor,
        texts: list[str] | None = None,
        text_strategy: str = "class_anchors",
        class_semantic_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode images (and optionally text) into multimodal features.

        Args:
            images: (B, 3, H, W).
            texts: List of B report strings, or None.
            text_strategy: How to handle missing text:
                "class_anchors", "vis2sem", or "visual_only".
            class_semantic_embeds: Pre-computed class anchor embeddings,
                shape (B, d_model). Used when text_strategy="class_anchors".

        Returns:
            Dict with keys: f_final, v_cls_proj, v_patches_proj, s_proj,
            v_guided, attn_weights.
        """
        device = images.device

        # Visual encoding
        v_cls, v_patches = self.visual_encoder(images)  # (B, d_visual), (B, P, d_visual)
        v_cls_proj, v_patches_proj = self.visual_proj(v_cls, v_patches)  # (B, d), (B, P, d)

        # Semantic encoding
        if texts is not None and any(t.strip() for t in texts):
            s_cls, _ = self.semantic_encoder(texts, device)  # (B, d_semantic)
            s_proj = self.semantic_proj(s_cls)  # (B, d_model)
        elif text_strategy == "vis2sem":
            s_proj = self.vis2sem(v_cls_proj)  # (B, d_model)
        elif text_strategy == "class_anchors" and class_semantic_embeds is not None:
            s_proj = class_semantic_embeds  # (B, d_model)
        else:
            # Visual-only fallback: use v_cls_proj as both visual and "semantic"
            s_proj = v_cls_proj  # (B, d_model)

        # SGAM: semantic guides visual attention
        v_guided, attn_weights = self.sgam(s_proj, v_patches_proj)  # (B, d), (B, P)

        # Gated fusion
        f_final = self.fusion(v_cls_proj, v_guided, s_proj)  # (B, d_model)

        return {
            "f_final": f_final,
            "v_cls_proj": v_cls_proj,
            "v_patches_proj": v_patches_proj,
            "s_proj": s_proj,
            "v_guided": v_guided,
            "attn_weights": attn_weights,
        }

    def forward_episode(
        self,
        support_images: torch.Tensor,
        support_texts: list[str] | None,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_texts: list[str] | None,
        n_way: int,
        class_semantic_embeds: torch.Tensor | None = None,
        text_strategy: str = "class_anchors",
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a single episode.

        Args:
            support_images: (n_way * k_shot, 3, H, W).
            support_texts: List of n_way * k_shot report strings, or None.
            support_labels: (n_way * k_shot,), values 0..n_way-1.
            query_images: (n_way * q_query, 3, H, W).
            query_texts: List of n_way * q_query report strings, or None.
            n_way: Number of classes.
            class_semantic_embeds: Optional class anchor embeddings (n_way, d_model).
            text_strategy: How to handle text at inference.

        Returns:
            Dict with keys: logits, support_out, query_out, prototypes.
        """
        # Encode support set
        # If using class anchors, expand to match support set size
        support_anchors = None
        if class_semantic_embeds is not None and support_texts is None:
            k_shot = support_images.shape[0] // n_way
            support_anchors = class_semantic_embeds.repeat_interleave(k_shot, dim=0)

        support_out = self.encode_multimodal(
            support_images, support_texts, text_strategy, support_anchors
        )

        # Compute prototypes
        prototypes = self.prototype(
            support_out["f_final"], support_labels, n_way, class_semantic_embeds
        )  # (n_way, d_model)

        # Encode query set
        query_anchors = None
        if class_semantic_embeds is not None and query_texts is None:
            # For queries, we don't know the class — use vis2sem or visual_only
            query_anchors = None

        query_out = self.encode_multimodal(
            query_images, query_texts, text_strategy, query_anchors
        )

        # Compute distances and logits
        distances = compute_distances(
            query_out["f_final"], prototypes, self.distance
        )  # (n_query, n_way)
        logits = -distances  # Higher = more similar

        return {
            "logits": logits,
            "support_out": support_out,
            "query_out": query_out,
            "prototypes": prototypes,
        }

    def forward(
        self,
        images: torch.Tensor,
        texts: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Standard forward pass (non-episodic, for Phase 1 alignment).

        Args:
            images: (B, 3, H, W).
            texts: List of B report strings.

        Returns:
            Dict with multimodal features.
        """
        return self.encode_multimodal(images, texts)
