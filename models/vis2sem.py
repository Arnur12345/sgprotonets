"""Visual-to-semantic auxiliary projection for text-free inference."""

import torch
import torch.nn as nn


class Vis2Sem(nn.Module):
    """Auxiliary MLP that predicts text embeddings from visual embeddings.

    Used at inference when per-image text is unavailable. Trained with
    MSE loss against actual semantic embeddings.

    Args:
        d_model: Shared embedding dimension.
        d_hidden: Hidden layer dimension.
    """

    def __init__(self, d_model: int, d_hidden: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, v_cls_proj: torch.Tensor) -> torch.Tensor:
        """Predict semantic embedding from visual features.

        Args:
            v_cls_proj: Projected visual [CLS] features, shape (B, d_model).

        Returns:
            Predicted semantic features, shape (B, d_model).
        """
        return self.mlp(v_cls_proj)
