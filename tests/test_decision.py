import math
import unittest

from backend.decision import classification_decision, validate_threshold


class DecisionTests(unittest.TestCase):
    labels = ["Acórdão", "ARE", "Despacho", "RE", "Sentença"]

    def test_below_threshold_abstains_without_rounding(self):
        result = classification_decision([0.59999, 0.10001, 0.1, 0.1, 0.1], self.labels, 0.60)
        self.assertTrue(result["requires_review"])
        self.assertEqual(result["label"], "Não classificado")
        self.assertEqual(result["suggested_label"], "Acórdão")

    def test_threshold_equality_is_accepted(self):
        result = classification_decision([0.6, 0.1, 0.1, 0.1, 0.1], self.labels, 0.60)
        self.assertFalse(result["requires_review"])

    def test_ensemble_decision_uses_mean_probabilities(self):
        folds = [[0.9, 0.025, 0.025, 0.025, 0.025], [0.2] * 5]
        mean = [sum(values) / len(folds) for values in zip(*folds)]
        self.assertTrue(classification_decision(mean, self.labels, 0.60)["requires_review"])

    def test_missing_threshold_requires_review(self):
        result = classification_decision([0.99, 0.0025, 0.0025, 0.0025, 0.0025], self.labels)
        self.assertTrue(result["requires_review"])
        self.assertIsNone(result["threshold"])

    def test_invalid_thresholds_fail(self):
        for value in [-0.1, 1.1, math.nan, math.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_threshold(value)

    def test_invalid_probabilities_fail(self):
        for values in [[0.1] * 5, [math.nan] * 5, [-0.2, 0.3, 0.3, 0.3, 0.3]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                classification_decision(values, self.labels, 0.60)


if __name__ == "__main__":
    unittest.main()
