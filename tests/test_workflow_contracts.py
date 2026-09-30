"""Integration contracts for the workflow methods kept on Pipeline."""
import importlib
import pickle
import unittest

import numpy as np
from numpy.testing import assert_allclose

from potage import Config, Pipeline, SoupConfig


def sample_pipeline():
    X = np.random.RandomState(72).normal(size=(100, 3))
    y = X[:, 0] * X[:, 1] + np.sin(2 * X[:, 2])
    config = Config(carrier_freqs=[1], fm_depths=[1], pm_depths=[1],
                    power_scales=[1], wm_center_pcts=[50], wm_width_mods=[1])
    return Pipeline(X, y, ['a', 'b', 'c'], config=config,
                    feature_types=['numeric'] * 3, fold_eval_count=None,
                    max_memory=None, max_candidates=None,
                    exclude_families={'squared', 'log', 'sqrt', 'cube',
                                      'tanh', 'reciprocal', 'rank'})


class WorkflowContracts(unittest.TestCase):
    def assert_replay(self, p, result):
        assert_allclose(p.transform(p.X), result.X, rtol=1e-10, atol=1e-10)
        loaded = pickle.loads(pickle.dumps(p))
        assert_allclose(loaded.transform(p.X[:2]), result.X[:2],
                        rtol=1e-10, atol=1e-10)
        self.assertEqual(loaded.stages[-1].output.source_names, result.source_names)

    def test_auto_run_full_progression_and_replay(self):
        p = sample_pipeline()
        result = p.auto_run(select=3, max_stage=5, early_stop_marginal=-1, patience=10)
        self.assertIn('wm', [s._step_type for s in p.stages])
        self.assert_replay(p, result)

    def test_carrier_sweep_replay_and_rewind(self):
        p = sample_pipeline()
        raw = p.raw(select=3)
        cp = p.checkpoint()
        result, meta = p.discover_carrier_space(raw, select=3, spaces=['linear', 'log'])
        self.assertEqual(sum(meta['winner_counts'].values()), len(result.names))
        self.assertEqual(meta['skipped'], {})
        self.assert_replay(p, result)
        p.rewind(cp)
        assert_allclose(p.transform(p.X), raw.X)

    def test_deepen_workflow_replays(self):
        p = sample_pipeline()
        raw = p.raw(select=3)
        result = p.deepen(raw, select=3, max_depth=2, n_carry=3,
                          marginal_carry_threshold=-1, early_stop_marginal=-1,
                          decay_stop=-1)
        self.assert_replay(p, result)

    def test_family_pruning_does_not_modify_parent(self):
        p = sample_pipeline()
        cp = p.checkpoint()
        config_freqs = list(p.config.carrier_freqs)
        result = p.prune_run(select=2, max_stage=1, n_seeds=2, subsample=.6)
        self.assertTrue(result['keep'])
        self.assertEqual(result['keep'], set(result['family_counts']))
        self.assertFalse(result['keep'] & result['exclude'])
        self.assertEqual(p.checkpoint(), cp)
        self.assertEqual(config_freqs, p.config.carrier_freqs)

    def test_legacy_stage_imports_and_config_alias(self):
        legacy = importlib.import_module('potage._stages')
        self.assertIs(legacy.Config, Config)
        self.assertIs(legacy.SoupConfig, SoupConfig)
        self.assertIs(Config, SoupConfig)
        for name in ('build_stage0_raw', 'build_stage1_am', 'build_stage2_static_carriers',
                     'build_stage_fields', 'build_probe_zones', 'build_field_pipeline',
                     'neuron_probe_groups', 'fft_freqs', 'calibrate_carrier_freqs',
                     'estimate_candidates', 'apply_stage'):
            self.assertTrue(callable(getattr(legacy, name)))
