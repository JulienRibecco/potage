import unittest

import numpy as np

from examples.realtime_race_probe import SCENARIO_FEATURE_NAMES
from examples.realtime_scenario_crossfit_probe import (
    INVARIANT_FEATURE_NAMES,
    add_invariant_features,
)


class RealtimeScenarioCrossfitProbeTest(unittest.TestCase):
    def test_invariant_features_have_expected_geometry_scaling(self):
        row = np.zeros((1, len(SCENARIO_FEATURE_NAMES)), dtype=float)
        row[0, 20] = 2.0
        row[0, 21] = 1.0
        row[0, 23] = 4.0
        row[0, 24] = 3.0
        row[0, 25] = 5.0
        transformed = add_invariant_features(row)
        self.assertEqual(len(INVARIANT_FEATURE_NAMES), transformed.shape[1])
        self.assertEqual(2.0, transformed[0, 26])
        self.assertEqual(3.0, transformed[0, 27])
        self.assertEqual(1.0, transformed[0, 28])
        self.assertEqual(2.0, transformed[0, 29])
        self.assertEqual(5.0 / 3.0, transformed[0, 30])


if __name__ == "__main__":
    unittest.main()
