from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def binary_auc(labels: Iterable[float], scores: Iterable[float]) -> float:
    labels_arr = np.asarray(list(labels), dtype=np.float32)
    scores_arr = np.asarray(list(scores), dtype=np.float32)
    if labels_arr.ndim != 1 or scores_arr.ndim != 1:
        raise ValueError("labels 和 scores 必须是一维序列。")
    if labels_arr.size == 0:
        return float("nan")

    positives = (labels_arr >= 0.5).astype(np.float32)
    negatives = 1.0 - positives
    pos_total = positives.sum()
    neg_total = negatives.sum()
    if pos_total == 0 or neg_total == 0:
        return float("nan")

    order = np.argsort(-scores_arr)
    sorted_labels = positives[order]

    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1.0 - sorted_labels)
    tps = np.concatenate([[0.0], tps])
    fps = np.concatenate([[0.0], fps])

    tpr = tps / pos_total
    fpr = fps / neg_total

    auc = np.trapz(tpr, fpr)
    return float(auc)


def precision_recall_f1(
    labels: Iterable[float],
    scores: Iterable[float],
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    labels_arr = (np.asarray(list(labels), dtype=np.float32) >= 0.5).astype(np.int32)
    scores_arr = np.asarray(list(scores), dtype=np.float32)
    preds = (scores_arr >= threshold).astype(np.int32)

    tp = int(np.logical_and(preds == 1, labels_arr == 1).sum())
    fp = int(np.logical_and(preds == 1, labels_arr == 0).sum())
    fn = int(np.logical_and(preds == 0, labels_arr == 1).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)
