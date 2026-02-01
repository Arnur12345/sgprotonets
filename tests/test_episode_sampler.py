"""Tests for episodic sampler."""

import torch
import pytest

from data.episode_sampler import EpisodeSampler, unpack_episode


class MockDataset:
    """Minimal dataset mock with class_indices and classes attributes."""

    def __init__(self, n_classes: int = 5, samples_per_class: int = 30):
        self.classes = [f"class_{i}" for i in range(n_classes)]
        self.class_indices = {}
        idx = 0
        for c in range(n_classes):
            self.class_indices[c] = list(range(idx, idx + samples_per_class))
            idx += samples_per_class


def test_sampler_yields_correct_count():
    """Sampler yields the requested number of episodes."""
    dataset = MockDataset(n_classes=5, samples_per_class=30)
    sampler = EpisodeSampler(dataset, n_way=5, k_shot=1, q_query=15, num_episodes=10)

    episodes = list(sampler)
    assert len(episodes) == 10


def test_episode_size():
    """Each episode has correct number of indices."""
    dataset = MockDataset(n_classes=5, samples_per_class=30)
    sampler = EpisodeSampler(dataset, n_way=3, k_shot=5, q_query=10, num_episodes=5)

    for episode in sampler:
        expected = 3 * 5 + 3 * 10  # n_way * k_shot + n_way * q_query
        assert len(episode) == expected


def test_deterministic_with_seed():
    """Same seed produces same episodes."""
    dataset = MockDataset(n_classes=5, samples_per_class=30)

    sampler1 = EpisodeSampler(dataset, n_way=5, k_shot=1, q_query=15, num_episodes=5, seed=42)
    sampler2 = EpisodeSampler(dataset, n_way=5, k_shot=1, q_query=15, num_episodes=5, seed=42)

    episodes1 = list(sampler1)
    episodes2 = list(sampler2)

    for e1, e2 in zip(episodes1, episodes2):
        assert e1 == e2


def test_different_seeds_differ():
    """Different seeds produce different episodes."""
    dataset = MockDataset(n_classes=5, samples_per_class=30)

    sampler1 = EpisodeSampler(dataset, n_way=5, k_shot=1, q_query=15, num_episodes=5, seed=42)
    sampler2 = EpisodeSampler(dataset, n_way=5, k_shot=1, q_query=15, num_episodes=5, seed=99)

    episodes1 = list(sampler1)
    episodes2 = list(sampler2)

    assert episodes1 != episodes2


def test_too_few_samples_raises():
    """Raise error when class has too few samples."""
    dataset = MockDataset(n_classes=5, samples_per_class=3)

    with pytest.raises(ValueError, match="need at least"):
        EpisodeSampler(dataset, n_way=5, k_shot=1, q_query=15, num_episodes=1)


def test_unpack_episode():
    """unpack_episode correctly splits and relabels."""
    n_way, k_shot, q_query = 3, 2, 5
    n_total = n_way * (k_shot + q_query)

    batch = {
        "image": torch.randn(n_total, 3, 224, 224),
        "report": [f"report_{i}" for i in range(n_total)],
        "label": torch.zeros(n_total, dtype=torch.long),  # Original labels don't matter
    }

    support, query, query_labels = unpack_episode(batch, n_way, k_shot, q_query)

    assert support["image"].shape[0] == n_way * k_shot
    assert query["image"].shape[0] == n_way * q_query
    assert len(support["report"]) == n_way * k_shot
    assert len(query["report"]) == n_way * q_query

    # Check episode labels are 0..n_way-1
    assert support["label"].min() == 0
    assert support["label"].max() == n_way - 1
    assert query_labels.min() == 0
    assert query_labels.max() == n_way - 1
