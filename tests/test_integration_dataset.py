from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conference_social_sensing.data.build_dataset import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.json"


def real_data_available() -> bool:
    if not DATA_CONFIG.is_file():
        return False
    config = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    return all(
        (PROJECT_ROOT / source["svg_dir"]).is_dir()
        and (PROJECT_ROOT / source["annotations_path"]).is_file()
        for source in config.get("sources", [])
    )


@unittest.skipUnless(real_data_available(), "Private source data is not available")
class RealDatasetIntegrationTests(unittest.TestCase):
    def test_full_two_batch_dataset_builds_without_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = build_dataset(
                DATA_CONFIG,
                output_dir_override=Path(temporary_directory),
                extract_images_override=False,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["photos"], 300)
            self.assertEqual(report["counts"]["detections"], 1721)
            self.assertEqual(report["counts"]["groups"], 346)
            self.assertEqual(report["counts"]["annotated_persons"], 819)
            self.assertEqual(report["validation"]["errors"], 0)


if __name__ == "__main__":
    unittest.main()
