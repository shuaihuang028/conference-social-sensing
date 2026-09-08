"""Dataset and train-only feature normalization for group-form modeling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FeatureStandardizer:
    """Standardize tabular features using statistics from the training split only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> FeatureStandardizer:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("values must be a non-empty two-dimensional array")
        self.mean_ = array.mean(axis=0)
        scale = array.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        return (np.asarray(values, dtype=np.float32) - self.mean_) / self.scale_


class GroupCropDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        project_root: Path,
        geometry_columns: Sequence[str],
        label_columns: Sequence[str],
        geometry_values: np.ndarray,
        transform,
    ) -> None:
        if len(frame) != len(geometry_values):
            raise ValueError(
                "frame and geometry_values must contain the same number of rows"
            )
        self.frame = frame.reset_index(drop=True)
        self.project_root = project_root
        self.geometry_columns = list(geometry_columns)
        self.label_columns = list(label_columns)
        self.geometry_values = np.asarray(geometry_values, dtype=np.float32)
        self.labels = self.frame[self.label_columns].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = self.project_root / str(row["crop_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return {
            "image": image,
            "geometry": torch.from_numpy(self.geometry_values[index]),
            "labels": torch.from_numpy(self.labels[index]),
            "group_id": str(row["group_id"]),
            "photo_id": str(row["photo_id"]),
        }
