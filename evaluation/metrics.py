"""Accuracy, confidence intervals, and per-class statistics.

Includes both single-label and multi-label metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Any


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


# =============================================================================
# Multi-Label Metrics
# =============================================================================


def multilabel_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive multi-label classification metrics.

    Args:
        predictions: Binary predictions, shape (N, L) where L is num labels.
        labels: Ground truth multi-hot, shape (N, L).
        class_names: Optional list of class names for per-label breakdown.

    Returns:
        Dict with metrics:
            - exact_match: Fraction of samples with all labels correct
            - hamming_loss: Fraction of wrong labels
            - micro_precision, micro_recall, micro_f1: Micro-averaged
            - macro_precision, macro_recall, macro_f1: Macro-averaged
            - sample_f1: Mean F1 per sample
            - per_label_metrics: Dict per label with precision, recall, f1
    """
    n_samples, n_labels = labels.shape

    # Exact match ratio (subset accuracy)
    exact_match = float(np.all(predictions == labels, axis=1).mean())

    # Hamming loss (fraction of wrong labels)
    hamming = float(np.mean(predictions != labels))

    # Per-sample metrics
    tp_per_sample = (predictions * labels).sum(axis=1)
    fp_per_sample = (predictions * (1 - labels)).sum(axis=1)
    fn_per_sample = ((1 - predictions) * labels).sum(axis=1)

    # Sample-wise precision, recall, F1
    sample_precision = tp_per_sample / (tp_per_sample + fp_per_sample + 1e-8)
    sample_recall = tp_per_sample / (tp_per_sample + fn_per_sample + 1e-8)
    sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall + 1e-8)

    # Handle edge cases (no positive predictions or labels)
    sample_f1 = np.nan_to_num(sample_f1, nan=0.0)
    mean_sample_f1 = float(sample_f1.mean())

    # Micro-averaged (global TP, FP, FN)
    micro_tp = float((predictions * labels).sum())
    micro_fp = float((predictions * (1 - labels)).sum())
    micro_fn = float(((1 - predictions) * labels).sum())

    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    # Per-label metrics (for macro averaging)
    per_label_metrics = {}
    label_precisions = []
    label_recalls = []
    label_f1s = []

    for l in range(n_labels):
        label_preds = predictions[:, l]
        label_truth = labels[:, l]

        tp = float((label_preds * label_truth).sum())
        fp = float((label_preds * (1 - label_truth)).sum())
        fn = float(((1 - label_preds) * label_truth).sum())
        tn = float(((1 - label_preds) * (1 - label_truth)).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Handle edge case: no positive samples in ground truth
        if label_truth.sum() == 0:
            recall = 0.0
            f1 = 0.0

        label_name = class_names[l] if class_names else str(l)
        per_label_metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(label_truth.sum()),
        }

        label_precisions.append(precision)
        label_recalls.append(recall)
        label_f1s.append(f1)

    # Macro-averaged
    macro_precision = float(np.mean(label_precisions))
    macro_recall = float(np.mean(label_recalls))
    macro_f1 = float(np.mean(label_f1s))

    return {
        "exact_match": exact_match,
        "hamming_loss": hamming,
        "sample_f1": mean_sample_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label_metrics": per_label_metrics,
    }


def multilabel_auc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute AUC-ROC metrics for multi-label classification.

    Args:
        probabilities: Predicted probabilities, shape (N, L).
        labels: Ground truth multi-hot, shape (N, L).
        class_names: Optional list of class names.

    Returns:
        Dict with:
            - micro_auc: Micro-averaged AUC
            - macro_auc: Macro-averaged AUC
            - per_label_auc: Dict mapping label to AUC
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {
            "micro_auc": 0.0,
            "macro_auc": 0.0,
            "per_label_auc": {},
            "error": "sklearn not installed",
        }

    n_labels = labels.shape[1]
    per_label_auc = {}
    valid_aucs = []

    for l in range(n_labels):
        label_probs = probabilities[:, l]
        label_truth = labels[:, l]

        # AUC requires both positive and negative samples
        if len(np.unique(label_truth)) > 1:
            auc = roc_auc_score(label_truth, label_probs)
            label_name = class_names[l] if class_names else str(l)
            per_label_auc[label_name] = float(auc)
            valid_aucs.append(auc)
        else:
            label_name = class_names[l] if class_names else str(l)
            per_label_auc[label_name] = None  # Undefined

    # Macro AUC (average over valid labels)
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # Micro AUC (flatten all predictions)
    # Only include labels that have both positive and negative samples
    valid_mask = np.array([
        len(np.unique(labels[:, l])) > 1 for l in range(n_labels)
    ])

    if valid_mask.any():
        flat_probs = probabilities[:, valid_mask].ravel()
        flat_labels = labels[:, valid_mask].ravel()
        micro_auc = float(roc_auc_score(flat_labels, flat_probs))
    else:
        micro_auc = 0.0

    return {
        "micro_auc": micro_auc,
        "macro_auc": macro_auc,
        "per_label_auc": per_label_auc,
    }


def confidence_interval_multilabel(
    metric_values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute mean and confidence interval for multi-label metrics.

    Args:
        metric_values: List of per-episode metric values (e.g., F1 scores).
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (mean, half-width of CI).
    """
    if len(metric_values) < 2:
        return float(np.mean(metric_values)) if metric_values else 0.0, 0.0

    n = len(metric_values)
    mean = np.mean(metric_values)
    se = stats.sem(metric_values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean), float(h)
