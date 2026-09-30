import unittest

from examples.realtime_latency_intervention import paired_intervention


class RealtimeLatencyInterventionTest(unittest.TestCase):
    def test_identical_latency_is_exactly_matched(self):
        result = paired_intervention(
            pairs=20,
            seed=7,
            focal=1,
            low_latency=2,
            high_latency=2,
        )
        self.assertEqual(0.0, result["paired_effect"])
        self.assertEqual(0, result["discordant_pairs"])


if __name__ == "__main__":
    unittest.main()
