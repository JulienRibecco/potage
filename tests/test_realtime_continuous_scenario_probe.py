import unittest

from examples.realtime_continuous_scenario_probe import (
    HELDOUT_SPEEDS,
    IN_SUPPORT_SPEEDS,
    SAFE_CELLS,
    generate_dataset,
)
from examples.realtime_race_probe import SCENARIO_FEATURE_NAMES


class RealtimeContinuousScenarioProbeTest(unittest.TestCase):
    def test_safe_cells_exclude_fixed_nodes(self):
        self.assertEqual(22, len(SAFE_CELLS))
        self.assertNotIn([0, 2], [cell.tolist() for cell in SAFE_CELLS])
        self.assertNotIn([2, 1], [cell.tolist() for cell in SAFE_CELLS])
        self.assertNotIn([4, 2], [cell.tolist() for cell in SAFE_CELLS])

    def test_dataset_has_scenario_columns_and_generation_counts(self):
        X, y, context, groups, generation = generate_dataset(16, 1201, IN_SUPPORT_SPEEDS)
        self.assertEqual(len(SCENARIO_FEATURE_NAMES), X.shape[1])
        self.assertEqual(4, context.shape[1])
        self.assertEqual(len(y), len(groups))
        self.assertEqual(16, generation["games_requested"])
        self.assertEqual(16, generation["games_kept"] + generation["draws"])

    def test_speed_regimes_are_distinct(self):
        self.assertTrue(set(HELDOUT_SPEEDS).isdisjoint(IN_SUPPORT_SPEEDS))


if __name__ == "__main__":
    unittest.main()
