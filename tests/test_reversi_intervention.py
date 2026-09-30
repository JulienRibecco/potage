import unittest

import numpy as np

from examples.reversi_intervention import (
    evaluate_scenarios,
    factor_components,
    make_scenarios,
)
from examples.reversi_probe import BLACK, initial_board, legal_moves


class ReversiInterventionTest(unittest.TestCase):
    def test_factor_components_are_finite(self):
        board = initial_board()
        move = legal_moves(board, BLACK)[0]
        components = factor_components(board, BLACK, move)

        self.assertEqual({"mobility", "safety", "combined"}, set(components))
        self.assertTrue(all(np.isfinite(value) for value in components.values()))

    def test_zero_strength_paired_score_is_exactly_neutral(self):
        scenarios = make_scenarios(4, seed=11, opening_plies=6)
        result = evaluate_scenarios(scenarios, "combined", 0.0)

        self.assertEqual(0.5, result["paired_score"])
        self.assertEqual(0.0, result["decision_change_rate"])


if __name__ == "__main__":
    unittest.main()
