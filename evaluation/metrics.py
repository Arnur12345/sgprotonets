"""Accuracy, confidence intervals, and per-class statistics."""

import numpy as np
from scipy import stats


def compute_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute mean accuracy.

    Args:
        predictions: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(np.mean(predictions == labels))


def confidence_interval(
    accuracies: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute mean and confidence interval over episode accuracies.

    Args:
        accuracies: List of per-episode accuracies.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (mean, half-width of CI).
    """
    n = len(accuracies)
    mean = np.mean(accuracies)
    se = stats.sem(accuracies)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean), float(h)


def per_class_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict[int, float]:
    """Compute per-class accuracy.

    Args:
        predictions: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).
        n_classes: Total number of classes.

    Returns:
        Dict mapping class index to accuracy.
    """
    result = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            result[c] = float(np.mean(predictions[mask] == labels[mask]))
        else:
            result[c] = 0.0
    return result
