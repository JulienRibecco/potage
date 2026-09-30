import unittest

from examples.realtime_stage_selection_probe import run_probe


class RealtimeStageSelectionProbeTest(unittest.TestCase):
    def test_recipe_selector_matrix_runs_on_grouped_split(self):
        result = run_probe(
            n_games=60,
            seed=12,
            select=6,
            recipes=("raw", "raw_am"),
            methods=("omp",),
        )
        self.assertEqual(2, len(result["results"]))
        self.assertEqual(
            {"raw", "raw_am"},
            {row["recipe"] for row in result["results"]},
        )
        self.assertTrue(all(row["test"]["brier"] >= 0 for row in result["results"]))
        self.assertNotEqual(
            result["split"]["discovery_games"],
            result["split"]["test_games"],
        )


if __name__ == "__main__":
    unittest.main()
