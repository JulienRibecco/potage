import unittest

from examples.realtime_scenario_effect_probe import collect_dataset


class RealtimeScenarioEffectTransportTest(unittest.TestCase):
    def test_collect_dataset_accepts_heldout_intervention_magnitudes(self):
        train = collect_dataset(12, 1401, "speed", range(2), speed_delta=1)
        test = collect_dataset(12, 1402, "speed", range(2), speed_delta=2)
        self.assertEqual(train[0].shape[1], test[0].shape[1])
        self.assertGreater(train[1].sum(), 0)
        self.assertGreaterEqual(test[1].sum(), 0)

    def test_position_steps_change_the_paired_dataset(self):
        one = collect_dataset(20, 1401, "initial_position", range(2), position_steps=1)
        two = collect_dataset(20, 1401, "initial_position", range(2), position_steps=2)
        self.assertEqual(one[0].shape[1], two[0].shape[1])
        self.assertEqual(len(one[1]), len(one[3]))
        self.assertEqual(len(two[1]), len(two[3]))
        self.assertTrue((one[1] >= 0).all())
        self.assertTrue((two[1] >= 0).all())


if __name__ == "__main__":
    unittest.main()
