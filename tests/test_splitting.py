import unittest

from conference_social_sensing.data.split_dataset import (
    allocate_block_counts,
    create_contiguous_block_split,
    create_source_aware_block_split,
)


class SplitDatasetTests(unittest.TestCase):
    def test_largest_remainder_allocates_every_block(self):
        counts = allocate_block_counts(15, {"train": 0.70, "val": 0.15, "test": 0.15})
        self.assertEqual(counts, {"train": 11, "val": 2, "test": 2})

    def test_contiguous_blocks_never_cross_splits(self):
        photo_ids = [f"CLEO_{number}" for number in range(401, 551)]
        result = create_contiguous_block_split(photo_ids, block_size=10)

        self.assertEqual(len(result), 150)
        self.assertEqual(result["photo_id"].nunique(), 150)
        self.assertTrue((result.groupby("scene_id")["split"].nunique() == 1).all())
        self.assertEqual(
            result["split"].value_counts().to_dict(),
            {"train": 110, "val": 20, "test": 20},
        )
        self.assertEqual(result.iloc[0]["photo_id"], "CLEO_401")
        self.assertEqual(result.iloc[-1]["photo_id"], "CLEO_550")

    def test_split_is_deterministic_for_unsorted_inputs(self):
        ordered = ["CLEO_1", "CLEO_2", "CLEO_10", "CLEO_11"]
        first = create_contiguous_block_split(ordered, block_size=2)
        second = create_contiguous_block_split(list(reversed(ordered)), block_size=2)
        self.assertEqual(first.to_dict("records"), second.to_dict("records"))

    def test_each_source_is_represented_in_every_split(self):
        import pandas as pd

        photos = pd.DataFrame(
            {
                "photo_id": [f"A_{index}" for index in range(1, 31)]
                + [f"B_{index}" for index in range(1, 31)],
                "source_batch": ["source_a"] * 30 + ["source_b"] * 30,
            }
        )
        result = create_source_aware_block_split(photos, block_size=5)
        self.assertTrue((result.groupby(["source_batch", "split"]).size() > 0).all())
        self.assertTrue((result.groupby("scene_id")["split"].nunique() == 1).all())
        self.assertEqual(result["scene_id"].nunique(), 12)


if __name__ == "__main__":
    unittest.main()
