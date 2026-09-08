import unittest

import numpy as np

from conference_social_sensing.data.build_group_crops import (
    expanded_union_bounds,
    geometry_features,
)


class GroupCropTests(unittest.TestCase):
    def test_expanded_union_is_clipped_to_image(self):
        boxes = np.array([[0, 10, 20, 40], [30, 20, 50, 60]])
        self.assertEqual(
            expanded_union_bounds(
                boxes, image_width=100, image_height=80, margin_ratio=0.20
            ),
            (0, 0, 60, 70),
        )

    def test_geometry_is_normalized(self):
        boxes = np.array([[0, 0, 20, 20], [20, 0, 40, 20]])
        features = geometry_features(boxes, image_width=100, image_height=50)
        self.assertAlmostEqual(features["geom_center_x"], 0.2)
        self.assertAlmostEqual(features["geom_center_y"], 0.2)
        self.assertAlmostEqual(features["geom_union_width"], 0.4)
        self.assertAlmostEqual(features["geom_union_height"], 0.4)
        self.assertGreater(features["geom_pair_distance_mean"], 0)
        self.assertTrue(all(0 <= value <= 1 for value in features.values()))


if __name__ == "__main__":
    unittest.main()
