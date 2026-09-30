import unittest

import numpy as np

from examples.realtime_race_probe import (
    FEATURE_NAMES,
    NODE_POSITIONS,
    move_one,
    play_game,
    state_features,
)


class RealtimeRaceProbeTest(unittest.TestCase):
    def test_movement_is_one_cell_per_tick(self):
        position = np.asarray([0, 0])
        target = NODE_POSITIONS[2]
        moved = move_one(position, target)
        self.assertEqual(1, int(np.abs(moved - position).sum()))
        self.assertLess(int(np.abs(moved - target).sum()), int(np.abs(position - target).sum()))

    def test_game_emits_timed_snapshots(self):
        X, y, context, groups = play_game(3, np.random.RandomState(4))
        self.assertEqual(18, len(X))
        self.assertTrue(all(len(row) == len(FEATURE_NAMES) for row in X))
        self.assertEqual(4, len(context[0]))
        self.assertEqual({3}, set(groups))


if __name__ == "__main__":
    unittest.main()
