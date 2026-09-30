import unittest

from examples.realtime_scenario_effect_probe import collect_dataset


class RealtimeScenarioEffectProbeTest(unittest.TestCase):
    def test_paired_effect_dataset_is_grouped_and_nontrivial(self):
        X, y, context, groups, generation = collect_dataset(
            24, 1301, "speed", range(3)
        )
        self.assertEqual(26, X.shape[1])
        self.assertEqual(4, context.shape[1])
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(groups))
        self.assertGreater(y.sum(), 0)
        self.assertEqual(24, generation["pairs_requested"])

    def test_position_intervention_has_same_snapshot_schema(self):
        X, y, context, groups, _ = collect_dataset(
            12, 1301, "initial_position", range(2)
        )
        self.assertEqual(26, X.shape[1])
        self.assertEqual(4, context.shape[1])
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(groups))


if __name__ == "__main__":
    unittest.main()
