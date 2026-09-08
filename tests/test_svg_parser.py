from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from conference_social_sensing.data.svg_parser import parse_svg


def _jpeg_data_url(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


class SvgParserTests(unittest.TestCase):
    def test_extracts_image_box_label_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            svg_path = Path(temporary_directory) / "TEST_1.svg"
            data_url = _jpeg_data_url(20, 30)
            svg_path.write_text(
                (
                    '<svg width="20" height="30" xmlns="http://www.w3.org/2000/svg" '
                    'xmlns:xlink="http://www.w3.org/1999/xlink">'
                    f'<image width="20" height="30" xlink:href="{data_url}" />'
                    '<rect x="1" y="2" width="10" height="20">'
                    "<title>conf:0.875</title></rect>"
                    '<text x="6" y="5">7</text>'
                    "</svg>"
                ),
                encoding="utf-8",
            )

            parsed, issues = parse_svg(svg_path)

            self.assertEqual(issues, [])
            self.assertEqual(parsed.photo["photo_id"], "TEST_1")
            self.assertEqual(parsed.photo["width"], 20)
            self.assertEqual(parsed.photo["height"], 30)
            self.assertEqual(len(parsed.detections), 1)
            detection = parsed.detections[0]
            self.assertEqual(detection["detection_id"], 7)
            self.assertEqual(detection["person_id"], "TEST_1__p7")
            self.assertEqual(detection["x2"], 11.0)
            self.assertAlmostEqual(detection["detector_confidence"], 0.875)


if __name__ == "__main__":
    unittest.main()
