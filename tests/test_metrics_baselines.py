import unittest

import numpy as np

from conference_social_sensing.baselines.metrics import multilabel_metrics
from conference_social_sensing.baselines.models import (
    MajorityCombinationBaseline,
    SizeOnlyBaseline,
)


class BaselineTests(unittest.TestCase):
    def test_multilabel_metrics(self):
        truth = np.array([[1, 0], [1, 1], [0, 1]])
        predicted = np.array([[1, 0], [1, 0], [0, 1]])
        metrics = multilabel_metrics(truth, predicted, ["A", "B"])

        self.assertAlmostEqual(metrics["exact_match_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["hamming_loss"], 1 / 6)
        self.assertAlmostEqual(metrics["micro_f1"], 6 / 7)
        self.assertAlmostEqual(metrics["macro_f1"], (1.0 + 2 / 3) / 2)

    def test_majority_predicts_most_common_combination(self):
        labels = np.array([[1, 0], [1, 0], [0, 1]])
        model = MajorityCombinationBaseline().fit(labels)
        predicted, probabilities = model.predict(2)

        np.testing.assert_array_equal(predicted, np.array([[1, 0], [1, 0]]))
        np.testing.assert_allclose(probabilities[0], np.array([2 / 3, 1 / 3]))

    def test_size_only_uses_global_fallback_and_never_predicts_empty(self):
        sizes = np.array([2, 2, 3, 3])
        labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        model = SizeOnlyBaseline(threshold=0.75).fit(sizes, labels)
        predicted, probabilities = model.predict([2, 3, 9])

        np.testing.assert_array_equal(predicted[0], np.array([1, 0]))
        np.testing.assert_array_equal(predicted[1], np.array([0, 1]))
        # Unseen size 9 has [0.5, 0.5]; argmax deterministically selects A.
        np.testing.assert_array_equal(predicted[2], np.array([1, 0]))
        np.testing.assert_allclose(probabilities[2], np.array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
