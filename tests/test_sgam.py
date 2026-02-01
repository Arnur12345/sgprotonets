"""Tests for Semantic-Guided Attention Module."""

import torch
import pytest

from models.sgam import SGAM


@pytest.fixture
def sgam():
    """SGAM in eval mode (no dropout) for deterministic tests."""
    module = SGAM(d_model=256, num_heads=4, dropout=0.0)
    module.eval()
    return module


def test_output_shape(sgam):
    """SGAM produces correct output shapes."""
    B, P, d = 4, 196, 256
    s_proj = torch.randn(B, d)
    v_patches = torch.randn(B, P, d)

    v_guided, attn_weights = sgam(s_proj, v_patches)

    assert v_guided.shape == (B, d)
    assert attn_weights.shape == (B, P)


def test_attention_weights_sum_to_one(sgam):
    """Attention weights should approximately sum to 1 per sample."""
    B, P, d = 2, 49, 256
    s_proj = torch.randn(B, d)
    v_patches = torch.randn(B, P, d)

    _, attn_weights = sgam(s_proj, v_patches)

    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=0.05)


def test_gradient_flow(sgam):
    """Gradients flow through SGAM."""
    B, P, d = 2, 196, 256
    s_proj = torch.randn(B, d, requires_grad=True)
    v_patches = torch.randn(B, P, d, requires_grad=True)

    v_guided, _ = sgam(s_proj, v_patches)
    loss = v_guided.sum()
    loss.backward()

    assert s_proj.grad is not None
    assert v_patches.grad is not None
    # Gradient flows via residual connections even if attention gradient is small
    assert s_proj.grad.abs().sum() > 0
    assert v_patches.grad.abs().sum() > 0


def test_batch_independence(sgam):
    """Each sample in batch processed independently."""
    P, d = 196, 256
    s1 = torch.randn(1, d)
    s2 = torch.randn(1, d)
    v1 = torch.randn(1, P, d)
    v2 = torch.randn(1, P, d)

    # Process individually
    out1, _ = sgam(s1, v1)
    out2, _ = sgam(s2, v2)

    # Process as batch
    s_batch = torch.cat([s1, s2], dim=0)
    v_batch = torch.cat([v1, v2], dim=0)
    out_batch, _ = sgam(s_batch, v_batch)

    assert torch.allclose(out1, out_batch[0:1], atol=1e-5)
    assert torch.allclose(out2, out_batch[1:2], atol=1e-5)
