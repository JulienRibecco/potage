import unittest

from examples.reversi_intervention_phases import run_phase_probe


class ReversiInterventionPhaseTest(unittest.TestCase):
    def test_phase_probe_preserves_requested_phase_order(self):
        result = run_phase_probe(
            pairs=2,
            opening_plies=[0, 8],
            strength=0.0,
            seed=5,
            bootstrap_repetitions=10,
        )

        self.assertEqual([0, 8], [row["opening_plies"] for row in result["phases"]])
        self.assertTrue(all(row["paired_score"] == 0.5 for row in result["phases"]))


if __name__ == "__main__":
    unittest.main()
