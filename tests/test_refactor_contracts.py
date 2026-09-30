import unittest
from unittest.mock import patch
import warnings
import numpy as np
from numpy.testing import assert_allclose
from potage import (SoupPipe, SoupConfig, FeaturePool, PrimitiveLibrary,
                    apply_stage, multi_pass, recipe_light)


class RefactorContracts(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(9).normal(size=(90, 3))
        self.y = self.X[:, 0] + .1 * self.X[:, 1]

    def pipe(self, **kw):
        return SoupPipe(self.X, self.y, ['mass', 'a', 'sin'],
                        fold_eval_count=None, max_memory=None, **kw)

    def test_standalone_am_reuses_reference_for_single_rows(self):
        kw = dict(stage='am', names=['x', 'y', 'z'], feature_types=['numeric'] * 3,
                  reference_X=self.X, include_input=False, filter_threshold=0,
                  dedup_threshold=0)
        whole = apply_stage(X=self.X[:4], **kw)
        one = apply_stage(X=self.X[:1], **kw)
        self.assertEqual(whole.names, one.names)
        assert_allclose(whole.X[:1], one.X)

    def test_classification_select_uses_the_same_one_vs_rest_dispatch(self):
        import potage._select as s
        p = SoupPipe(self.X, np.arange(90) % 3, ['x', 'y', 'z'],
                     task='classification', fold_eval_count=None, max_memory=None)
        raw = p.raw(select=None)
        expected = s.per_class_select(PrimitiveLibrary(raw.X, raw.names, raw._families),
                                      p.y, max_features=2)
        actual = p.select(raw, select=2)
        self.assertEqual(set(actual.names), {n for h in expected.values() for n in h.selected_names()})
        self.assertIsNotNone(p.stages[-1]._per_class_histories)

    def test_correlation_is_not_reported_or_diagnosed_as_r2(self):
        p = self.pipe()
        p.raw(select=3, method='correlation')
        self.assertIsNone(p.stages[-1].final_r2)
        self.assertIsNone(p.stages[-1].diagnosis)
        self.assertEqual(p.stages[-1].history.score_kind, 'mrmr')
        self.assertNotIn('R²=', repr(p.stages[-1]))

    def test_rewind_preserves_a_preexisting_handle_and_budget(self):
        p = self.pipe()
        raw = p.raw(select=3)
        budget = p._candidates_used
        p.fuse(raw)
        p.rewind()
        self.assertEqual(p._candidates_used, budget)
        p.carriers(raw, select=2)

    def test_failed_stage_does_not_charge_budget(self):
        p = self.pipe()
        raw = p.raw(select=3)
        before = p.checkpoint()
        with self.assertRaises(ValueError):
            p.carriers(raw, select=2, feature_space='not-a-space')
        self.assertEqual(p.checkpoint(), before)

    def test_invalid_sweep_configuration_is_not_swallowed(self):
        p = self.pipe()
        raw = p.raw(select=3)
        with self.assertRaises(ValueError):
            p.discover_carrier_space(raw, spaces=['not-a-space'])

    def test_multipass_budget_override_and_empty_input_contract(self):
        pool = FeaturePool(self.X, self.y, ['x', 'y', 'z'], max_candidates=500,
                           fold_eval_count=None, max_memory=None)
        self.assertEqual(pool.build().max_candidates, 500)
        with self.assertRaisesRegex(ValueError, 'pass|columns'):
            multi_pass(self.X[:, :2], self.y, ['x', 'y'], recipe_light())

    def test_multipass_sources_are_exact_not_substrings(self):
        excluded = {'squared', 'log', 'sqrt', 'cube', 'tanh', 'reciprocal', 'rank'}
        pool = FeaturePool(self.X, self.X[:, 0], ['mass', 'a', 'sin'],
                           min_cols=1, fold_eval_count=None, max_memory=None,
                           exclude_families=excluded)
        p = pool.build()
        raw = p.raw(select=1)
        sr = pool.capture(raw, p)
        self.assertEqual(raw.names, ['mass'])
        self.assertEqual(sr.used, {'mass'})
        carrier = p.carriers(raw, select=1)
        self.assertEqual(pool.capture(carrier, p).used, {'mass'})

    def test_representative_policy_is_shared_by_dedup_methods(self):
        X = np.c_[self.X[:, 0], 100 * self.X[:, 0]]
        lib = PrimitiveLibrary(X, ['simple', 'complex'], ['raw'] * 2, complexities=[0, 5])
        for method in ('hierarchical', 'union-find'):
            self.assertEqual(lib.dedup_correlated(method=method, representative='complexity').names,
                             ['simple'])
        with self.assertRaises(ValueError):
            lib.dedup_correlated(method='typo')

    def test_native_omp_fallback_does_not_require_native_binary(self):
        from potage._accel import c_omp_select
        lib = PrimitiveLibrary(self.X, ['x', 'y', 'z'], ['raw'] * 3)
        with patch('potage._accel.HAS_C_SOUP', False):
            self.assertGreater(len(c_omp_select(lib, self.y, max_features=2)), 0)

    def test_budget_guards_and_rewind_restore_the_full_state(self):
        from potage import CandidateBudgetExceeded, MemoryBudgetExceeded
        p = self.pipe(max_candidates=1)
        before = p.checkpoint()
        with self.assertRaises(CandidateBudgetExceeded):
            p.raw(select=3)
        self.assertEqual(before, p.checkpoint())
        p.max_candidates = 10000
        raw = p.raw(select=3)
        cp = p.checkpoint()
        p.carriers(raw, select=2)
        p.rewind(cp)
        self.assertEqual(cp, p.checkpoint())
        self.assertEqual(p._candidate_families_seen, set(cp.families))
        p.max_memory = 1
        with self.assertRaises(MemoryBudgetExceeded):
            p.am(raw, select=2)
        self.assertEqual(cp, p.checkpoint())

    def test_ancestry_and_replay_survive_every_core_builder_and_pickle(self):
        import pickle
        config = SoupConfig(carrier_freqs=[1], fm_depths=[1], pm_depths=[1],
                            wm_center_pcts=[50], wm_width_mods=[1],
                            power_scales=[1], field_pct_levels=[50], field_scales=[.3])
        for operation in ('am', 'carriers', 'fft_carriers', 'fm', 'wm', 'fm_nonsmooth', 'fields'):
            with self.subTest(operation=operation):
                p = self.pipe(config=config)
                raw = p.raw(select=3)
                if operation in ('fm', 'wm', 'fm_nonsmooth'):
                    fs = getattr(p, operation)(raw, raw, select=2)
                else:
                    fs = getattr(p, operation)(raw, select=2)
                self.assertTrue(fs.names)
                self.assertTrue(all(s and s <= {'mass', 'a', 'sin'} for s in fs.source_names))
                assert_allclose(p.transform(self.X), fs.X, atol=1e-10)
                restored = pickle.loads(pickle.dumps(p))
                self.assertEqual(restored.stages[-1].output.source_names, fs.source_names)
                assert_allclose(restored.transform(self.X[:1]), fs.X[:1], atol=1e-10)
                restored.rewind()
                restored.carriers(restored.stages[-1].output, select=1)

    def test_cross_am_matches_shared_builder_and_exact_ancestry(self):
        from potage._stages import build_stage1_am
        from potage._cross import _build_cross_am
        from potage import FeatureSet
        left = FeatureSet(self.X[:, :1], ['mass'], ['numeric'])
        right = FeatureSet(self.X[:, 1:], ['a', 'sin'], ['numeric'] * 2)
        config = SoupConfig(power_scales=[1])
        cols, names, families = _build_cross_am((left, right), config)
        whole, all_names, all_fams = build_stage1_am(self.X, ['mass','a','sin'],
                                                     [True]*3, config)
        lookup = dict(zip(all_names, whole))
        for name, col in zip(names, cols):
            assert_allclose(col, lookup[name])
            self.assertIn('mass', name.sources)
            self.assertEqual(len(name.sources), 2)

    def test_omp_history_records_gains_separately_from_ranking(self):
        from potage import residual_select
        lib = PrimitiveLibrary(self.X, ['x','y','z'], ['raw']*3)
        h = residual_select(lib, self.y, max_features=3)
        assert_allclose(np.cumsum(h.marginal_r2()), h.cumulative_r2())
        self.assertTrue(all('selection_score' in step for step in h.steps))

    def test_deepen_replays_a_subset_carry(self):
        p = self.pipe(config=SoupConfig(carrier_freqs=[1], fm_depths=[1],
                                       pm_depths=[1], wm_center_pcts=[50]))
        raw = p.raw(select=3)
        result = p.deepen(raw, select=2, max_depth=1, n_carry=2,
                          marginal_carry_threshold=-1, early_stop_marginal=-1)
        assert_allclose(p.transform(self.X), result.X, atol=1e-10)

    def test_memory_estimators_and_config_ownership(self):
        config = SoupConfig(carrier_freqs=[1])
        p = self.pipe(config=config)
        p.config.carrier_freqs.append(2)
        self.assertEqual(config.carrier_freqs, [1])
        raw = p.raw(select=3)
        for stage, args in [('raw',()), ('am',(raw,)), ('fields',(raw,)),
                            ('carriers',(raw,)), ('fm',(raw,raw)), ('wm',(raw,raw))]:
            self.assertGreater(p.estimate_memory(stage, *args)['peak_bytes'], 0)
        self.assertGreater(p.predict_pipeline_memory()['total_peak_bytes'], 0)

    def test_public_selection_arguments_fail_before_clipping(self):
        p = self.pipe()
        raw = p.raw(select=3)
        cp = p.checkpoint()
        for count in (-1, 1.5, 100.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                p.select(raw, select=count)
            self.assertEqual(cp, p.checkpoint())
        with self.assertRaises(ValueError):
            p.raw(select=None, method='typo')
        self.assertEqual(cp, p.checkpoint())

    def test_classification_with_explicit_validation_routes_consistently(self):
        p = SoupPipe(self.X, np.arange(90) % 3, ['x','y','z'],
                     task='classification', X_val=self.X[:30],
                     y_val=np.arange(30) % 3, max_memory=None)
        raw = p.raw(select=2)
        p.select(raw, select=1)
        self.assertEqual(p.stages[-1].history.score_kind, 'class_union')
        self.assertIsNone(p.stages[-1].final_r2)
        self.assertEqual(len(p.stages[-1]._per_class_histories), 3)

    def test_automatic_r2_stopping_rejects_correlation_scores(self):
        p = self.pipe()
        with self.assertRaisesRegex(ValueError, 'R²'):
            p.auto_run(method='correlation')
        raw = p.raw(select=3)
        with self.assertRaisesRegex(ValueError, 'R²'):
            p.deepen(raw, method='correlation')

    def test_multipass_replay_preserves_exact_sources(self):
        import pickle
        excluded = {'squared', 'log', 'sqrt', 'cube', 'tanh', 'reciprocal', 'rank'}
        X, _ = np.linalg.qr(self.X - self.X.mean(axis=0))
        y = X[:, 0] + .1 * X[:, 1]
        result = multi_pass(X, y, ['mass','a','sin'],
                            lambda p: p.raw(select=1), k=2, min_cols=1,
                            exclude_families=excluded, fold_eval_count=None,
                            max_memory=None)
        self.assertEqual(result.pass_names, [['mass'], ['a']])
        self.assertEqual(result.unused_names, ['sin'])
        restored = pickle.loads(pickle.dumps(result))
        assert_allclose(restored.transform(X), result.X)

    def test_native_config_is_rejected_rather_than_ignored(self):
        from potage._accel import build_lib_c
        with self.assertRaisesRegex(ValueError, 'Config'):
            build_lib_c(self.X, self.X[:2], ['x','y','z'], config=SoupConfig())
