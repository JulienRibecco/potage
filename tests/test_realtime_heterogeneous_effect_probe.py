import unittest

import numpy as np

from examples.realtime_heterogeneous_effect_probe import collect_dataset
from examples.realtime_race_probe import FEATURE_NAMES


class RealtimeHeterogeneousEffectProbeTest(unittest.TestCase):
    def test_matched_effect_dataset_has_grouped_phase_rows(self):
        X, y, context, groups, generation = collect_dataset(
            pairs=30,
            seed=5,
            phases=(0, 4, 8),
        )
        self.assertEqual(len(FEATURE_NAMES), X.shape[1])
        self.assertEqual(4, context.shape[1])
        self.assertEqual(len(y), len(groups))
        self.assertGreaterEqual(float(y.mean()), 0.0)
        self.assertLessEqual(float(y.mean()), 1.0)
        self.assertGreater(generation["rows"], 0)
        self.assertEqual({10, 50, 90}, set(np.unique(X[:, 0] * 100).astype(int)))


if __name__ == "__main__":
    unittest.main()
