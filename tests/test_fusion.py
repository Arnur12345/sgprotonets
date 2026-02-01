"""Tests for gated fusion module."""

import torch
import pytest

from models.fusion import GatedFusion


@pytest.fixture
def fusion():
    return GatedFusion(d_model=256, dropout=0.0)


def test_output_shape(fusion):
    """Fusion output has correct shape."""
    B, d = 4, 256
    v_cls = torch.randn(B, d)
    v_guided = torch.randn(B, d)
    s_proj = torch.randn(B, d)

    f_final = fusion(v_cls, v_guided, s_proj)
    assert f_final.shape == (B, d)


def test_gradient_flow(fusion):
    """Gradients flow through all three inputs."""
    B, d = 2, 256
    v_cls = torch.randn(B, d, requires_grad=True)
    v_guided = torch.randn(B, d, requires_grad=True)
    s_proj = torch.randn(B, d, requires_grad=True)

    f_final = fusion(v_cls, v_guided, s_proj)
    loss = f_final.sum()
    loss.backward()

    assert v_cls.grad is not None and v_cls.grad.abs().sum() > 0
    assert v_guided.grad is not None and v_guided.grad.abs().sum() > 0
    assert s_proj.grad is not None and s_proj.grad.abs().sum() > 0


def test_deterministic_in_eval(fusion):
    """Fusion is deterministic in eval mode."""
    fusion.eval()
    B, d = 2, 256
    v_cls = torch.randn(B, d)
    v_guided = torch.randn(B, d)
    s_proj = torch.randn(B, d)

    out1 = fusion(v_cls, v_guided, s_proj)
    out2 = fusion(v_cls, v_guided, s_proj)

    assert torch.allclose(out1, out2)


def test_different_inputs_different_outputs(fusion):
    """Different inputs produce different outputs."""
    fusion.eval()
    d = 256
    v_cls = torch.randn(1, d)
    v_guided_a = torch.randn(1, d)
    v_guided_b = torch.randn(1, d)
    s_proj = torch.randn(1, d)

    out_a = fusion(v_cls, v_guided_a, s_proj)
    out_b = fusion(v_cls, v_guided_b, s_proj)

    assert not torch.allclose(out_a, out_b, atol=1e-4)
