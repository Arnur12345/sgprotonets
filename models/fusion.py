"""Gated fusion module for combining visual and semantic features."""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Gated fusion of semantically-guided visual and semantic features.

    Gate: g = sigmoid(W_gate * [v_guided; s])
    Fused: g * v_guided + (1 - g) * s
    Final: MLP([v_cls; fused]) -> d_model

    Args:
        d_model: Shared embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Final MLP: combines v_cls with fused representation
        self.final_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        v_cls_proj: torch.Tensor,
        v_guided: torch.Tensor,
        s_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal features.

        Args:
            v_cls_proj: Projected visual [CLS], shape (B, d_model).
            v_guided: SGAM output, shape (B, d_model).
            s_proj: Projected semantic [CLS], shape (B, d_model).

        Returns:
            f_final: Fused multimodal representation, shape (B, d_model).
        """
        # Gated combination of v_guided and s_proj
        concat = torch.cat([v_guided, s_proj], dim=-1)  # (B, 2*d_model)
        g = self.gate(concat)  # (B, d_model)
        fused = g * v_guided + (1 - g) * s_proj  # (B, d_model)

        # Combine with visual CLS
        combined = torch.cat([v_cls_proj, fused], dim=-1)  # (B, 2*d_model)
        f_final = self.final_mlp(combined)  # (B, d_model)
        return f_final
