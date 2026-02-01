"""ViT wrapper that extracts [CLS] token and patch token sequence."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class VisualEncoder(nn.Module):
    """Visual encoder wrapping a ViT backbone.

    Extracts both the [CLS] global feature and the full patch token sequence
    (spatial features needed by SGAM).

    Args:
        cfg: Model config with visual_encoder settings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        enc_cfg = cfg.visual_encoder

        if enc_cfg.type == "biomedclip":
            self._init_biomedclip(enc_cfg.name)
        else:
            self._init_hf_vit(enc_cfg.name)

        # Freeze/unfreeze
        if enc_cfg.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Optionally unfreeze last N blocks
            if enc_cfg.unfreeze_last_n_blocks > 0:
                self._unfreeze_last_n(enc_cfg.unfreeze_last_n_blocks)

    def _init_biomedclip(self, model_name: str) -> None:
        """Initialize from BiomedCLIP (open_clip)."""
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:" + model_name
        )
        self.encoder = model.visual.trunk
        self._encoder_type = "timm"

    def _init_hf_vit(self, model_name: str) -> None:
        """Initialize from HuggingFace ViT."""
        from transformers import ViTModel

        self.encoder = ViTModel.from_pretrained(model_name)
        self._encoder_type = "hf"

    def _unfreeze_last_n(self, n: int) -> None:
        """Unfreeze the last N transformer blocks."""
        if self._encoder_type == "timm":
            blocks = list(self.encoder.blocks.children())
        else:
            blocks = list(self.encoder.encoder.layer)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract visual features.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            Tuple of:
                v_cls: [CLS] token features, shape (B, d_visual).
                v_patches: Patch token sequence, shape (B, num_patches, d_visual).
        """
        if self._encoder_type == "timm":
            return self._encode_timm(x)
        else:
            return self._encode_hf(x)

    def _encode_timm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode using timm-style ViT (BiomedCLIP)."""
        # Forward through patch embed + blocks
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)
        x = self.encoder.norm_pre(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        v_cls = x[:, 0]  # (B, d_visual)
        v_patches = x[:, 1:]  # (B, num_patches, d_visual)
        return v_cls, v_patches

    def _encode_hf(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode using HuggingFace ViT."""
        outputs = self.encoder(pixel_values=x)
        v_cls = outputs.last_hidden_state[:, 0]  # (B, d_visual)
        v_patches = outputs.last_hidden_state[:, 1:]  # (B, num_patches, d_visual)
        return v_cls, v_patches

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Alias for encode()."""
        return self.encode(x)
