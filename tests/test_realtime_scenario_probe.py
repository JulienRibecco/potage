import unittest

import numpy as np

from examples.realtime_race_probe import (
    SCENARIO_FEATURE_NAMES,
    play_game,
)


class RealtimeScenarioProbeTest(unittest.TestCase):
    def test_speed_and_initial_state_features_are_emitted(self):
        X, y, context, groups = play_game(
            4,
            np.random.RandomState(8),
            start_positions={1: np.asarray([2, 0]), -1: np.asarray([4, 4])},
            speed_override={1: 2, -1: 1},
            scenario_features=True,
        )
        self.assertEqual(len(SCENARIO_FEATURE_NAMES), len(X[0]))
        self.assertEqual(2.0, X[0][20])
        self.assertEqual(1.0, X[0][21])
        self.assertEqual(1.0, X[0][22])
        self.assertEqual(4, len(context[0]))
        self.assertEqual({4}, set(groups))


if __name__ == "__main__":
    unittest.main()
