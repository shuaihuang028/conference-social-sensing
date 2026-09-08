import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from conference_social_sensing.modeling.group_model import VisualGeometryGroupClassifier
from conference_social_sensing.modeling.predictor import GroupFormPredictor


class PredictorTests(unittest.TestCase):
    def _checkpoint(self, directory: Path) -> Path:
        model = VisualGeometryGroupClassifier(
            num_geometry_features=2,
            num_labels=4,
            hidden_dim=8,
            dropout=0.0,
            pretrained=False,
            freeze_backbone=True,
        )
        path = directory / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_columns": ["form_A", "form_B", "form_C", "form_D"],
                "geometry_columns": ["group_size", "distance"],
                "geometry_mean": [2.0, 0.5],
                "geometry_scale": [1.0, 0.25],
                "config": {
                    "hidden_dim": 8,
                    "dropout": 0.0,
                    "freeze_backbone": True,
                    "image_size": 64,
                    "threshold": 0.5,
                },
            },
            path,
        )
        return path

    def test_checkpoint_can_predict_and_returns_nonempty_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            predictor = GroupFormPredictor(
                self._checkpoint(Path(temporary)), device="cpu"
            )
            result = predictor.predict(
                Image.new("RGB", (80, 80), "white"),
                {"group_size": 2.0, "distance": 0.5},
            )
        self.assertTrue(result["labels"])
        self.assertEqual(set(result["probabilities"]), {"A", "B", "C", "D"})
        self.assertTrue(
            all(
                0 <= probability <= 1
                for probability in result["probabilities"].values()
            )
        )

    def test_missing_geometry_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            predictor = GroupFormPredictor(
                self._checkpoint(Path(temporary)), device="cpu"
            )
            with self.assertRaisesRegex(ValueError, "Missing geometry"):
                predictor.predict(
                    Image.new("RGB", (80, 80), "white"),
                    {"group_size": 2.0},
                )


if __name__ == "__main__":
    unittest.main()
