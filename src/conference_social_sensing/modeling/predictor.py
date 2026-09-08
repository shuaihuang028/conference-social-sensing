"""Reusable checkpoint loader and inference interface for group-form prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from conference_social_sensing.modeling.group_model import VisualGeometryGroupClassifier


def _evaluation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GroupFormPredictor:
    """Load one training checkpoint and expose stable single/batch inference."""

    def __init__(self, checkpoint_path: Path | str, *, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = _auto_device() if device == "auto" else torch.device(device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.label_columns = list(checkpoint["label_columns"])
        self.geometry_columns = list(checkpoint["geometry_columns"])
        self.geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
        self.geometry_scale = np.asarray(checkpoint["geometry_scale"], dtype=np.float32)
        self.config = dict(checkpoint["config"])
        self.threshold = float(self.config.get("threshold", 0.5))
        self.image_size = int(self.config.get("image_size", 224))
        self.transform = _evaluation_transform(self.image_size)

        self.model = VisualGeometryGroupClassifier(
            num_geometry_features=len(self.geometry_columns),
            num_labels=len(self.label_columns),
            hidden_dim=int(self.config["hidden_dim"]),
            dropout=float(self.config["dropout"]),
            pretrained=False,
            freeze_backbone=bool(self.config.get("freeze_backbone", True)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

    def _geometry_array(
        self, geometry: Mapping[str, float] | Sequence[float]
    ) -> np.ndarray:
        if isinstance(geometry, Mapping):
            missing = [name for name in self.geometry_columns if name not in geometry]
            if missing:
                raise ValueError(f"Missing geometry features: {missing}")
            values = np.asarray(
                [geometry[name] for name in self.geometry_columns], dtype=np.float32
            )
        else:
            values = np.asarray(geometry, dtype=np.float32)
        if values.shape != (len(self.geometry_columns),):
            raise ValueError(
                f"Expected {len(self.geometry_columns)} geometry values, got shape {values.shape}"
            )
        return (values - self.geometry_mean) / self.geometry_scale

    @staticmethod
    def _open_image(image: Image.Image | Path | str) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as source:
            return source.convert("RGB")

    @torch.inference_mode()
    def predict_batch(
        self,
        images: Sequence[Image.Image | Path | str],
        geometries: Sequence[Mapping[str, float] | Sequence[float]],
        *,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if len(images) != len(geometries):
            raise ValueError("images and geometries must have equal length")
        if not images:
            return []
        selected_threshold = self.threshold if threshold is None else float(threshold)
        if not 0 <= selected_threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        image_tensor = torch.stack(
            [self.transform(self._open_image(image)) for image in images]
        ).to(self.device)
        geometry_tensor = torch.from_numpy(
            np.stack([self._geometry_array(geometry) for geometry in geometries])
        ).to(self.device)
        probabilities = (
            torch.sigmoid(self.model(image_tensor, geometry_tensor)).cpu().numpy()
        )
        predictions = probabilities >= selected_threshold
        empty_rows = np.where(predictions.sum(axis=1) == 0)[0]
        if len(empty_rows):
            predictions[empty_rows, probabilities[empty_rows].argmax(axis=1)] = True

        results = []
        for probability_row, prediction_row in zip(probabilities, predictions):
            labels = [
                label.removeprefix("form_")
                for label, selected in zip(self.label_columns, prediction_row)
                if selected
            ]
            results.append(
                {
                    "labels": labels,
                    "probabilities": {
                        label.removeprefix("form_"): float(probability)
                        for label, probability in zip(
                            self.label_columns, probability_row
                        )
                    },
                    "threshold": selected_threshold,
                }
            )
        return results

    def predict(
        self,
        image: Image.Image | Path | str,
        geometry: Mapping[str, float] | Sequence[float],
        *,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.predict_batch([image], [geometry], threshold=threshold)[0]
