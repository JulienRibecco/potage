import pickle
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from potage import SoupPipe, SoupConfig, PrimitiveLibrary, residual_select
from potage._stages import build_stage1_am
from potage._kfold import _kfold_omp, _make_folds

UNARY = {'squared', 'log', 'sqrt', 'cube', 'tanh', 'reciprocal', 'rank'}


class ReplayTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(13)
        self.X = rng.normal(size=(100, 2))
        self.y = self.X[:, 0] * self.X[:, 1]

    def pipe(self, **kwargs):
        return SoupPipe(self.X, self.y, ['a', 'b'], feature_types=['numeric'] * 2,
                        fold_eval_count=None, max_memory=None, **kwargs)

    def assert_replays(self, pipe, fs):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            assert_allclose(pipe.transform(self.X), fs.X, atol=1e-10)
            samples = np.vstack([self.X[:5], [[-20, 50]]])
            whole = pipe.transform(samples)
            rows = np.vstack([pipe.transform(row[None, :]) for row in samples])
            assert_allclose(rows, whole, atol=1e-10)
            assert_allclose(pipe.transform(samples[::-1])[::-1], whole, atol=1e-10)
            restored = pickle.loads(pickle.dumps(pipe))
            assert_allclose(restored.transform(samples), whole, atol=1e-10)
            selected = restored.select(restored.stages[-1].output, select=2)
            self.assertEqual(restored.transform(samples).shape[1], selected.X.shape[1])

    def test_raw_ties_use_the_same_empirical_cdf_at_fit_and_transform(self):
        X = np.repeat(np.arange(10), 3)[:, None] / 100
        p = SoupPipe(X, np.arange(len(X)), ['x'], feature_types=['numeric'],
                     fold_eval_count=None, max_memory=None,
                     exclude_families=(UNARY - {'rank'}) | {'raw'})
        raw = p.raw(select=None)
        assert_allclose(raw.X, p.transform(X))
        assert_array_equal(raw.X[:3, 0], [0, 0, 0])
        assert_allclose(p.transform([[-1], [1]])[:, 0], [0, 1])

    def test_single_set_am_is_independent_of_batch_and_preserves_validation(self):
        p = self.pipe(exclude_families=UNARY, X_val=self.X[:7])
        raw = p.raw(select=None)
        am = p.am(raw, select=None)
        self.assertTrue({'am_ratio', 'am_power', 'am_logratio'} <= set(am._families))
        assert_allclose(am._X_val, p.transform(self.X[:7]))
        self.assert_replays(p, am)

    def test_power_exponents_clip_to_training_range(self):
        train = np.array([[2., 0.], [3., 10.], [4., 5.]])
        cols, names, _ = build_stage1_am(
            np.array([[2., 100.], [2., -100.]]), ['a', 'b'], [True, True],
            config=SoupConfig(power_scales=[2.]), reference_X=train)
        assert_allclose(cols[names.index('a^(2·b)')], [4., 1.], atol=1e-9)

    def test_cross_am_and_modulation_replay(self):
        for method in ('am', 'fm', 'wm', 'fm_nonsmooth'):
            with self.subTest(method=method):
                p = self.pipe(exclude_families=UNARY, X_val=self.X[:7])
                raw = p.raw(select=None)
                carrier = p.carriers(raw, select=2)
                fs = getattr(p, method)(raw, carrier, select=None)
                self.assertGreater(fs.X.shape[1], 0)
                assert_allclose(fs._X_val, p.transform(self.X[:7]), atol=1e-10)
                self.assert_replays(p, fs)

    def test_fractional_category_codes_have_distinct_names(self):
        X = np.repeat([.1, .2, .3], 10)[:, None]
        p = SoupPipe(X, np.arange(30), ['c'], feature_types=['categorical'],
                     fold_eval_count=None, max_memory=None)
        fs = p.raw(select=None)
        self.assertEqual(len(fs.names), len(set(fs.names)))
        assert_allclose(fs.X, p.transform(X))


class SelectionScoringTests(unittest.TestCase):
    def test_omp_fold_scores_equal_independent_validation_refits(self):
        rng = np.random.RandomState(22)
        X = rng.normal(size=(60, 12)) + 3
        y = 50 + rng.normal(size=60)
        folds = _make_folds(60, 3, 42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            h = _kfold_omp(X, list(map(str, range(12))), ['raw'] * 12, y, 5, folds)
        selected = []
        for step in h.steps:
            selected.append(int(step['name']))
            scores = []
            for val in folds:
                tr = np.setdiff1d(np.arange(len(X)), val)
                model = Ridge(alpha=1e-6).fit(X[tr][:, selected], y[tr])
                scores.append(r2_score(y[val], model.predict(X[val][:, selected])))
            self.assertAlmostEqual(step['cumulative_r2'], np.mean(scores), places=9)
            self.assertAlmostEqual(step['fold_r2_std'], np.std(scores), places=9)

    def test_explicit_validation_omp_scores_validation_without_selecting_on_it(self):
        rng = np.random.RandomState(22)
        X = rng.normal(size=(60, 8)) + 3
        y = 50 + X[:, 0] + rng.normal(size=60)
        V = rng.normal(size=(20, 8)) + 4
        vy = 70 + rng.normal(size=20)
        lib = PrimitiveLibrary(X, list(map(str, range(8))), ['raw'] * 8, X_val=V)
        h = residual_select(lib, y, max_features=3, y_val=vy)
        train_h = residual_select(lib, y, max_features=3)
        self.assertEqual(h.selected_names(), train_h.selected_names())
        for k, step in enumerate(h.steps, 1):
            sel = list(map(int, h.selected_names()[:k]))
            expected = Ridge(alpha=1e-6).fit(X[:, sel], y).predict(V[:, sel])
            self.assertAlmostEqual(step['cumulative_r2'], r2_score(vy, expected), places=8)

    def test_pipeline_select_honors_fold_scoring(self):
        rng = np.random.RandomState(29)
        X = rng.normal(size=(60, 3))
        y = 10 + rng.normal(size=60)
        p = SoupPipe(X, y, ['a', 'b', 'c'], feature_types=['numeric'] * 3,
                     fold_eval_count=3, max_memory=None, exclude_families=UNARY)
        raw = p.raw(select=None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            selected = p.select(raw, select=2)
        h = p.stages[-1].history
        self.assertIn('fold_r2_std', h.steps[-1])
        cols = [raw.names.index(n) for n in selected.names]
        scores = []
        for val in _make_folds(60, 3, 42):
            tr = np.setdiff1d(np.arange(60), val)
            model = Ridge(alpha=1e-6).fit(raw.X[tr][:, cols], y[tr])
            scores.append(r2_score(y[val], model.predict(raw.X[val][:, cols])))
        self.assertAlmostEqual(h.steps[-1]['cumulative_r2'], np.mean(scores), places=9)

    def test_invalid_fold_counts_fail_before_scoring(self):
        for count in (0, 1, 7, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _make_folds(12, count, 42)
