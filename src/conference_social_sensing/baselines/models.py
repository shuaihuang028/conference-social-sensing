"""Interpretable baselines for multi-label group-form prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _ensure_nonempty(predictions: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    result = predictions.copy()
    empty_rows = np.where(result.sum(axis=1) == 0)[0]
    if len(empty_rows):
        result[empty_rows, probabilities[empty_rows].argmax(axis=1)] = 1
    return result


@dataclass
class MajorityCombinationBaseline:
    """Predict the most frequent exact label combination in the training set."""

    combination_: np.ndarray | None = None
    prevalence_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> MajorityCombinationBaseline:
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 2 or len(labels) == 0:
            raise ValueError("y must be a non-empty two-dimensional array")
        combinations, counts = np.unique(labels, axis=0, return_counts=True)
        max_count = counts.max()
        tied = combinations[counts == max_count]
        # Lexicographic tie-break keeps the result deterministic.
        self.combination_ = tied[np.lexsort(tied[:, ::-1].T)[0]].astype(int)
        self.prevalence_ = labels.mean(axis=0)
        return self

    def predict(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self.combination_ is None or self.prevalence_ is None:
            raise RuntimeError("fit must be called before predict")
        predictions = np.tile(self.combination_, (num_samples, 1))
        probabilities = np.tile(self.prevalence_, (num_samples, 1))
        return predictions, probabilities


@dataclass
class SizeOnlyBaseline:
    """Estimate label prevalence by observed group size."""

    threshold: float = 0.5
    prevalence_by_size_: dict[int, np.ndarray] | None = None
    global_prevalence_: np.ndarray | None = None

    def fit(self, group_sizes: Sequence[int], y: np.ndarray) -> SizeOnlyBaseline:
        sizes = np.asarray(group_sizes, dtype=int)
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 2 or len(labels) == 0 or len(sizes) != len(labels):
            raise ValueError(
                "group_sizes and y must contain the same non-zero number of rows"
            )
        self.global_prevalence_ = labels.mean(axis=0)
        self.prevalence_by_size_ = {
            int(size): labels[sizes == size].mean(axis=0) for size in np.unique(sizes)
        }
        return self

    def predict(self, group_sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        if self.prevalence_by_size_ is None or self.global_prevalence_ is None:
            raise RuntimeError("fit must be called before predict")
        sizes = np.asarray(group_sizes, dtype=int)
        probabilities = (
            np.vstack(
                [
                    self.prevalence_by_size_.get(int(size), self.global_prevalence_)
                    for size in sizes
                ]
            )
            if len(sizes)
            else np.empty((0, len(self.global_prevalence_)))
        )
        predictions = (probabilities >= self.threshold).astype(int)
        predictions = _ensure_nonempty(predictions, probabilities)
        return predictions, probabilities


def decode_labels(row: np.ndarray, label_names: Sequence[str]) -> str:
    return ";".join(
        name.removeprefix("form_") for name, value in zip(label_names, row) if value
    )
