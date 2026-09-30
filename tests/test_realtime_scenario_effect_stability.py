import unittest

from examples.realtime_scenario_effect_stability import consensus_counts


class RealtimeScenarioEffectStabilityTest(unittest.TestCase):
    def test_consensus_counts_deduplicate_features_per_run(self):
        result = consensus_counts([
            ["a", "a", "b"],
            ["a", "c"],
            ["b", "c"],
        ])
        by_name = {row["name"]: row for row in result}
        self.assertEqual(2, by_name["a"]["runs"])
        self.assertEqual(2, by_name["b"]["runs"])
        self.assertEqual(2, by_name["c"]["runs"])
        self.assertEqual(2 / 3, by_name["a"]["fraction"])


if __name__ == "__main__":
    unittest.main()
