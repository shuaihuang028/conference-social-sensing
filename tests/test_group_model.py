import unittest

import numpy as np
import torch

from conference_social_sensing.modeling.group_dataset import FeatureStandardizer
from conference_social_sensing.modeling.group_model import VisualGeometryGroupClassifier


class GroupModelTests(unittest.TestCase):
    def test_standardizer_uses_safe_scale_for_constant_columns(self):
        values = np.array([[1.0, 2.0], [1.0, 4.0]], dtype=np.float32)
        standardizer = FeatureStandardizer().fit(values)
        transformed = standardizer.transform(values)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(transformed[:, 0], np.zeros(2))

    def test_model_output_shape_and_frozen_backbone(self):
        model = VisualGeometryGroupClassifier(
            num_geometry_features=3,
            num_labels=4,
            pretrained=False,
            freeze_backbone=True,
        )
        output = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.backbone.parameters())
        )
        self.assertTrue(
            any(parameter.requires_grad for parameter in model.classifier.parameters())
        )


if __name__ == "__main__":
    unittest.main()
