from __future__ import annotations

import unittest

from conference_social_sensing.data.annotations import (
    parse_group_forms,
    validate_soft_distribution,
)


class GroupFormParsingTests(unittest.TestCase):
    def test_single_label(self) -> None:
        self.assertEqual(parse_group_forms("B"), ("B",))

    def test_multilabel_delimiters_are_canonicalized(self) -> None:
        for raw in ("A; D", "D,A", "A / D", "A + D", "D and A"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_group_forms(raw), ("A", "D"))

    def test_invalid_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_group_forms("A; E")


class SoftDistributionTests(unittest.TestCase):
    def test_valid_sparse_distribution(self) -> None:
        valid, message = validate_soft_distribution(
            {"W": 0.7, "EA": 0.3, "O": None}, max_active_categories=3
        )
        self.assertTrue(valid)
        self.assertIsNone(message)

    def test_distribution_must_sum_to_one(self) -> None:
        valid, message = validate_soft_distribution(
            {"1": 0.4, "2": 0.4, "3": None}, max_active_categories=2
        )
        self.assertFalse(valid)
        self.assertIn("sum", message or "")

    def test_too_many_active_categories_is_rejected(self) -> None:
        valid, message = validate_soft_distribution(
            {"1": 0.3, "2": 0.3, "3": 0.4}, max_active_categories=2
        )
        self.assertFalse(valid)
        self.assertIn("active categories", message or "")


if __name__ == "__main__":
    unittest.main()
