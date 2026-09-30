import unittest

import numpy as np

from examples.connect_four_probe import (
    COLS,
    PLAYER_ONE,
    drop_piece,
    immediate_wins,
    legal_columns,
    line_length,
)


class ConnectFourProbeTest(unittest.TestCase):
    def test_vertical_win_and_full_columns(self):
        board = np.zeros((6, COLS), dtype=np.int8)
        for _ in range(3):
            board, _ = drop_piece(board, PLAYER_ONE, 2)
        self.assertEqual(1, len(immediate_wins(board, PLAYER_ONE)))
        board, row = drop_piece(board, PLAYER_ONE, 2)
        self.assertEqual(4, line_length(board, row, 2, PLAYER_ONE))
        for _ in range(2):
            board, _ = drop_piece(board, -PLAYER_ONE, 2)
        self.assertNotIn(2, legal_columns(board))

    def test_snapshot_features_have_expected_width(self):
        from examples.connect_four_probe import FEATURE_NAMES, state_features

        board = np.zeros((6, COLS), dtype=np.int8)
        self.assertEqual(len(FEATURE_NAMES), len(state_features(board, PLAYER_ONE)))


if __name__ == "__main__":
    unittest.main()
