import unittest

import numpy as np

from examples.reversi_probe import (
    BLACK,
    FEATURE_NAMES,
    apply_move,
    bit,
    bit_count,
    initial_board,
    legal_moves,
    play_game,
)


class ReversiEngineTest(unittest.TestCase):
    def test_opening_moves_and_flip(self):
        board = initial_board()
        moves = legal_moves(board, BLACK)

        self.assertEqual(
            {(2, 3), (3, 2), (4, 5), (5, 4)},
            {divmod(position, 8) for position, _ in moves},
        )

        move = next(move for move in moves if move[0] == 2 * 8 + 3)
        after = apply_move(board, BLACK, move)
        self.assertEqual((4, 1), (bit_count(after[0]), bit_count(after[1])))
        self.assertTrue(after[0] & bit(3, 3))

    def test_complete_game_emits_consistent_snapshots(self):
        X, y, context, groups = play_game(7, np.random.RandomState(1))

        self.assertEqual(5, len(X))
        self.assertTrue(all(len(row) == len(FEATURE_NAMES) for row in X))
        self.assertEqual({0.0, 1.0}, set(y))
        self.assertEqual({7}, set(groups))
        self.assertTrue(all(len(row) == 4 for row in context))


if __name__ == "__main__":
    unittest.main()
