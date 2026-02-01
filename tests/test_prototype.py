"""Tests for prototype computation and distance functions."""

import torch
import pytest

from models.prototypes import PrototypeComputation, compute_distances


@pytest.fixture
def proto_vanilla():
    return PrototypeComputation(mode="vanilla")


@pytest.fixture
def proto_weighted():
    return PrototypeComputation(mode="semantic_weighted")


def test_vanilla_shape(proto_vanilla):
    """Vanilla prototypes have correct shape."""
    n_way, k_shot, d = 5, 3, 256
    features = torch.randn(n_way * k_shot, d)
    labels = torch.arange(n_way).repeat_interleave(k_shot)

    protos = proto_vanilla(features, labels, n_way)
    assert protos.shape == (n_way, d)


def test_vanilla_is_mean(proto_vanilla):
    """Vanilla prototype equals mean of support features per class."""
    n_way, k_shot, d = 3, 2, 64
    features = torch.randn(n_way * k_shot, d)
    labels = torch.arange(n_way).repeat_interleave(k_shot)

    protos = proto_vanilla(features, labels, n_way)

    for c in range(n_way):
        expected = features[labels == c].mean(dim=0)
        assert torch.allclose(protos[c], expected, atol=1e-6)


def test_semantic_weighted_shape(proto_weighted):
    """Semantic-weighted prototypes have correct shape."""
    n_way, k_shot, d = 5, 3, 256
    features = torch.randn(n_way * k_shot, d)
    labels = torch.arange(n_way).repeat_interleave(k_shot)
    class_sem = torch.randn(n_way, d)

    protos = proto_weighted(features, labels, n_way, class_sem)
    assert protos.shape == (n_way, d)


def test_semantic_weighted_different_from_vanilla():
    """Semantic-weighted prototypes differ from vanilla when given diverse features."""
    n_way, k_shot, d = 3, 5, 64
    features = torch.randn(n_way * k_shot, d)
    labels = torch.arange(n_way).repeat_interleave(k_shot)
    class_sem = torch.randn(n_way, d)

    vanilla = PrototypeComputation("vanilla")(features, labels, n_way)
    weighted = PrototypeComputation("semantic_weighted")(
        features, labels, n_way, class_sem
    )

    # Should not be identical (extremely unlikely with random data)
    assert not torch.allclose(vanilla, weighted, atol=1e-4)


def test_cosine_distance_shape():
    """Cosine distance has correct shape."""
    n_query, n_way, d = 15, 5, 256
    queries = torch.randn(n_query, d)
    protos = torch.randn(n_way, d)

    dists = compute_distances(queries, protos, "cosine")
    assert dists.shape == (n_query, n_way)


def test_euclidean_distance_shape():
    """Euclidean distance has correct shape."""
    n_query, n_way, d = 15, 5, 256
    queries = torch.randn(n_query, d)
    protos = torch.randn(n_way, d)

    dists = compute_distances(queries, protos, "euclidean")
    assert dists.shape == (n_query, n_way)


def test_cosine_distance_self_is_zero():
    """Cosine distance to self should be ~0."""
    d = 256
    x = torch.randn(3, d)
    dists = compute_distances(x, x, "cosine")
    diagonal = torch.diag(dists)
    assert torch.allclose(diagonal, torch.zeros(3), atol=1e-5)


def test_gradient_through_prototypes(proto_vanilla):
    """Gradients flow through prototype computation."""
    n_way, k_shot, d = 3, 2, 64
    features = torch.randn(n_way * k_shot, d, requires_grad=True)
    labels = torch.arange(n_way).repeat_interleave(k_shot)

    protos = proto_vanilla(features, labels, n_way)
    loss = protos.sum()
    loss.backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0
