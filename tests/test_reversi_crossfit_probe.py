import unittest

import numpy as np

from examples.reversi_crossfit_probe import (
    ingredients_in_expression,
    split_development_test,
)


class ReversiCrossfitProbeTest(unittest.TestCase):
    def test_ingredient_parser_respects_full_feature_names(self):
        ingredients = ingredients_in_expression(
            "our_mobility/(1+|opponent_potential_mobility|)"
        )
        self.assertEqual(
            ["our_mobility", "opponent_potential_mobility"], ingredients
        )

    def test_development_test_split_is_game_disjoint(self):
        groups = np.repeat(np.arange(50), 3)
        development, test = split_development_test(groups, seed=9)
        self.assertFalse(set(development) & set(test))
        self.assertEqual(set(range(50)), set(development) | set(test))


if __name__ == "__main__":
    unittest.main()
