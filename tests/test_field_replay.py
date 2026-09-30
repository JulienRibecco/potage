"""Field candidate eligibility must depend on training rows, not replay batches."""
import unittest
import numpy as np
from potage import Config, Pipeline
from potage._fields import build_stage_fields


class FieldReplayTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[0., 0.], [0., 1.], [.5, .5], [1., 0.], [1., 1.]])
        self.validation = np.array([[.5, .5], [.51, .51], [.49, .49]])
        self.config = Config(field_pct_levels=[50], field_scales=[.2, .6])

    def build(self, X, reference=None):
        return build_stage_fields(X, ['x', 'y'], np.ones(2, dtype=bool),
                                  self.config, reference_X=reference)

    def test_constant_replay_zones_keep_training_columns_and_values(self):
        _, train_names, train_families = self.build(self.train)
        columns, names, families = self.build(self.validation, self.train)
        self.assertEqual(names, train_names)
        self.assertEqual(families, train_families)
        self.assertIn('field_zone', families)
        for col, family in zip(columns, families):
            if family in ('field_zone', 'field_zcount'):
                np.testing.assert_array_equal(col, np.ones(3))
        for i, row in enumerate(self.validation):
            single, single_names, _ = self.build(row[None, :], self.train)
            self.assertEqual(single_names, names)
            np.testing.assert_allclose(np.column_stack(single)[0], np.column_stack(columns)[i])

    def test_pipeline_validation_and_single_row_replay_agree(self):
        pipe = Pipeline(self.train, np.array([0., 0., 1., 0., 0.]), ['x', 'y'],
                        X_val=self.validation, y_val=np.ones(3),
                        feature_types=['numeric', 'numeric'], config=self.config,
                        fold_eval_count=None, max_memory=None,
                        exclude_families={'squared','log','sqrt','cube','tanh','reciprocal','rank'})
        raw = pipe.raw(select=None)
        final = pipe.fields(raw, select=None)
        replay = pipe.transform(self.validation)
        self.assertGreater(replay.shape[1], 0)
        np.testing.assert_allclose(replay, final._X_val)
        np.testing.assert_allclose(pipe.transform(self.train), final.X)
        for i, row in enumerate(self.validation):
            np.testing.assert_allclose(pipe.transform(row[None, :])[0], replay[i])


if __name__ == '__main__':
    unittest.main()
