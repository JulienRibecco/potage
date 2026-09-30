import unittest

import numpy as np

from potage import SoupPipe


class ExcludedFamilyDedupTest(unittest.TestCase):
    def test_excluded_transform_cannot_remove_allowed_raw_feature(self):
        bounded = np.linspace(0.05, 0.95, 200)
        X = bounded[:, None]
        y = bounded.copy()
        excluded_unary = {
            "squared",
            "log",
            "sqrt",
            "cube",
            "tanh",
            "reciprocal",
            "rank",
        }

        pipe = SoupPipe(
            X,
            y,
            ["bounded"],
            feature_types=["numeric"],
            fold_eval_count=None,
            exclude_families=excluded_unary,
        )
        raw = pipe.raw(select=None)

        self.assertEqual(raw.names, ["bounded"])
        self.assertEqual(raw._families, ["raw"])


if __name__ == "__main__":
    unittest.main()
