"""Visual and semantic projection heads into shared embedding space."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class VisualProjection(nn.Module):
    """Project visual features into shared embedding space.

    Architecture: d_visual → d_hidden → d_model with LayerNorm + GELU.
    Applied to both [CLS] and patch tokens.

    Args:
        cfg: Model config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_visual = cfg.d_visual
        d_hidden = cfg.d_hidden
        d_model = cfg.d_model

        self.cls_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(
        self, v_cls: torch.Tensor, v_patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project visual features.

        Args:
            v_cls: [CLS] features, shape (B, d_visual).
            v_patches: Patch features, shape (B, num_patches, d_visual).

        Returns:
            Tuple of projected v_cls (B, d_model) and v_patches (B, num_patches, d_model).
        """
        v_cls_proj = self.cls_proj(v_cls)  # (B, d_model)
        v_patches_proj = self.patch_proj(v_patches)  # (B, num_patches, d_model)
        return v_cls_proj, v_patches_proj


class SemanticProjection(nn.Module):
    """Project semantic features into shared embedding space.

    Architecture: d_semantic → d_hidden → d_model with LayerNorm + GELU.

    Args:
        cfg: Model config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_semantic = cfg.d_semantic
        d_hidden = cfg.d_hidden
        d_model = cfg.d_model

        self.proj = nn.Sequential(
            nn.Linear(d_semantic, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, s_cls: torch.Tensor) -> torch.Tensor:
        """Project semantic [CLS] features.

        Args:
            s_cls: Semantic [CLS] features, shape (B, d_semantic).

        Returns:
            Projected features, shape (B, d_model).
        """
        return self.proj(s_cls)  # (B, d_model)
