import json
import unittest
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from conference_social_sensing.api.app import create_app

GEOMETRY = {
    "group_size": 2,
    "geom_center_x": 0.5,
    "geom_center_y": 0.5,
    "geom_union_width": 0.3,
    "geom_union_height": 0.4,
    "geom_union_area": 0.12,
    "geom_mean_member_width": 0.1,
    "geom_mean_member_height": 0.2,
    "geom_center_std_x": 0.05,
    "geom_center_std_y": 0.02,
    "geom_pair_distance_mean": 0.1,
    "geom_pair_distance_max": 0.1,
}


class StubPredictor:
    device = "cpu"
    label_columns = ("form_A", "form_B", "form_C", "form_D")

    def predict(self, image, geometry, threshold=None):
        return {
            "labels": ["B"],
            "probabilities": {"A": 0.1, "B": 0.8, "C": 0.2, "D": 0.1},
            "threshold": 0.5 if threshold is None else threshold,
        }


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="JPEG")
    return stream.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(predictor=StubPredictor()))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labels"], ["A", "B", "C", "D"])

    def test_predict_group(self):
        response = self.client.post(
            "/predict-group",
            files={"image": ("crop.jpg", image_bytes(), "image/jpeg")},
            data={"geometry_json": json.dumps(GEOMETRY), "threshold": "0.6"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["labels"], ["B"])
        self.assertEqual(payload["threshold"], 0.6)
        self.assertGreaterEqual(payload["latency_ms"], 0)

    def test_rejects_non_image_upload(self):
        response = self.client.post(
            "/predict-group",
            files={"image": ("crop.txt", b"not an image", "text/plain")},
            data={"geometry_json": json.dumps(GEOMETRY)},
        )
        self.assertEqual(response.status_code, 415)

    def test_rejects_invalid_geometry(self):
        invalid = dict(GEOMETRY)
        invalid.pop("group_size")
        response = self.client.post(
            "/predict-group",
            files={"image": ("crop.jpg", image_bytes(), "image/jpeg")},
            data={"geometry_json": json.dumps(invalid)},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
