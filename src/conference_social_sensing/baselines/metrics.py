"""Dependency-light multi-label classification metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, object]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be two-dimensional arrays with equal shape"
        )
    if true.shape[1] != len(label_names):
        raise ValueError("label_names length must match the number of label columns")

    tp = ((true == 1) & (pred == 1)).sum(axis=0)
    fp = ((true == 0) & (pred == 1)).sum(axis=0)
    fn = ((true == 1) & (pred == 0)).sum(axis=0)

    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    per_label = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(true[:, index].sum()),
        }
        for index, name in enumerate(label_names)
    }
    return {
        "num_samples": int(true.shape[0]),
        "exact_match_accuracy": float(np.all(true == pred, axis=1).mean())
        if len(true)
        else 0.0,
        "hamming_loss": float(np.not_equal(true, pred).mean()) if len(true) else 0.0,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "per_label": per_label,
    }
