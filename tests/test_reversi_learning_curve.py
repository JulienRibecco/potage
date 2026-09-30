import unittest

import numpy as np

from examples.reversi_learning_curve import fixed_game_split, rows_for_games


class ReversiLearningCurveTest(unittest.TestCase):
    def test_fixed_game_split_is_disjoint_and_complete(self):
        groups = np.repeat(np.arange(20), 5)
        train, validation, test = fixed_game_split(groups, seed=7)

        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        self.assertEqual(set(range(20)), set(train) | set(validation) | set(test))
        self.assertEqual(20, len(rows_for_games(groups, test)))


if __name__ == "__main__":
    unittest.main()
