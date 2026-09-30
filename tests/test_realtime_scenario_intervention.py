import unittest

from examples.realtime_scenario_intervention import paired_intervention


class RealtimeScenarioInterventionTest(unittest.TestCase):
    def test_speed_intervention_returns_paired_outcomes(self):
        result = paired_intervention(pairs=20, seed=13, focal=1, kind="speed")
        self.assertGreater(result["pairs_kept"], 0)
        self.assertEqual(result["pairs_kept"], len(result["_deltas"]))
        self.assertGreaterEqual(result["treated_win_rate"], 0.0)
        self.assertLessEqual(result["treated_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
