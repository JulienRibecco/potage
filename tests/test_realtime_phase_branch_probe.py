import unittest

from examples.realtime_phase_branch_probe import collect_dataset
from examples.realtime_race_probe import FEATURE_NAMES


class RealtimePhaseBranchProbeTest(unittest.TestCase):
    def test_branches_share_their_prefix_state(self):
        X, y, context, groups, generation = collect_dataset(
            pairs=12,
            seed=9,
            phases=(0, 4, 8),
        )
        self.assertEqual(len(FEATURE_NAMES), X.shape[1])
        self.assertEqual(4, context.shape[1])
        self.assertEqual(len(y), len(groups))
        self.assertEqual(0, generation["common_state_mismatches"])
        self.assertGreater(generation["branches_kept"], 0)


if __name__ == "__main__":
    unittest.main()
