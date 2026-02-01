"""Semantic-Guided Attention Module (SGAM).

Cross-attention where semantic embedding queries visual patch tokens,
producing a semantically-attended visual feature.
"""

import torch
import torch.nn as nn


class SGAM(nn.Module):
    """Semantic-Guided Attention Module.

    Uses cross-attention with semantic embedding as Query and visual
    patch tokens as Keys/Values. This is the core contribution:
    language tells vision where to look.

    Args:
        d_model: Shared embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        s_proj: torch.Tensor,
        v_patches_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute semantically-guided visual feature.

        Args:
            s_proj: Projected semantic embedding, shape (B, d_model).
            v_patches_proj: Projected patch tokens, shape (B, num_patches, d_model).

        Returns:
            Tuple of:
                v_guided: Guided visual feature, shape (B, d_model).
                attn_weights: Attention weights, shape (B, num_patches).
        """
        # Expand semantic to query sequence of length 1: (B, 1, d_model)
        query = s_proj.unsqueeze(1)

        # Cross-attention: semantic queries, visual patches as keys/values
        # key/value: (B, num_patches, d_model)
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=v_patches_proj,
            value=v_patches_proj,
        )  # attn_out: (B, 1, d_model), attn_weights: (B, 1, num_patches)

        # Residual + norm
        attn_out = self.norm(attn_out + query)

        # FFN + residual + norm
        ffn_out = self.ffn(attn_out)
        out = self.norm2(ffn_out + attn_out)

        v_guided = out.squeeze(1)  # (B, d_model)
        attn_weights = attn_weights.squeeze(1)  # (B, num_patches)

        return v_guided, attn_weights
