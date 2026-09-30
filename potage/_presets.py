"""Multi-pass soup via FeaturePool / StageResult.

Diverse feature selection by running a user-defined recipe K times,
each pass excluding the base descriptors consumed by prior passes.

Usage::

    pool = FeaturePool(X, y, names, exclude_families={'reciprocal'},
                       fold_eval_count=3)

    passes = []
    for _ in range(3):
        p = pool.build()
        raw = p.raw(select=15)
        fs = p.select(p.fuse(raw, p.carriers(raw, select=15)), select=15)
        passes.append(pool.capture(fs, p))
        pool = pool.subtract(passes[-1])

    result = MultiPassResult.from_passes(passes, names)

    X_test_soup = result.transform(X_test)
    X_test_full = result.transform_with_unused(X_test)
"""

from dataclasses import dataclass, field
from copy import copy

import numpy as np


# ---------------------------------------------------------------------------
# StageResult — output of one pass, knows which base vars it consumed
# ---------------------------------------------------------------------------

class StageResult:
    """Result of one soup pass.

    Attributes
    ----------
    X : ndarray (n, d_selected)
        Selected feature matrix for this pass.
    names : list of str
        Selected feature names.
    used : set of str
        Exact raw descriptor names consumed by the selected features.
    pipe : SoupPipe
        Fitted pipeline (for transform).
    cols : list of int
        Column indices (into original X) this pass operated on.
    """

    __slots__ = ('X', 'names', 'used', 'pipe', 'cols')

    def __init__(self, fs, pipe, cols, all_names):
        self.X = fs.X
        self.names = list(fs.names)
        self.pipe = pipe
        self.cols = list(cols)
        self.used = set().union(*fs.source_names)


# ---------------------------------------------------------------------------
# FeaturePool — column pool with build / subtract
# ---------------------------------------------------------------------------

class FeaturePool:
    """Mutable-ish feature pool that builds SoupPipes and tracks consumption.

    Parameters
    ----------
    X : ndarray (n, d)
        Raw feature matrix.
    y : ndarray (n,)
        Target vector.
    names : list of str
        Feature names, length d.
    min_cols : int
        Minimum columns to allow a new pass (default 3).
    **pipe_kw
        Forwarded to SoupPipe constructor (config, exclude_families,
        fold_eval_count, force_numeric, max_candidates, ...).
    """

    def __init__(self, X, y, names, *, min_cols=3, force_numeric=True,
                 **pipe_kw):
        self._X = X
        self._y = y
        self._all_names = list(names)
        self._active = list(range(len(names)))
        if not isinstance(min_cols, int) or min_cols < 1:
            raise ValueError("min_cols must be a positive integer")
        self._min_cols = min_cols
        self._force_numeric = force_numeric
        # Extract X_val so build() can slice its columns in sync with X.
        self._X_val = pipe_kw.pop('X_val', None)
        self._pipe_kw = pipe_kw

    # -- public API --

    @property
    def n_active(self):
        """Number of active (non-consumed) columns."""
        return len(self._active)

    @property
    def active_names(self):
        """Names of active columns."""
        return [self._all_names[i] for i in self._active]

    @property
    def exhausted(self):
        """True if too few columns remain for another pass."""
        return len(self._active) < self._min_cols

    def build(self):
        """Create a fresh SoupPipe from active columns.

        Returns
        -------
        Pipeline
        """
        from ._pipe import Pipeline

        cols = self._active
        names = [self._all_names[i] for i in cols]
        ft = (np.full(len(cols), 'numeric', dtype='<U11')
              if self._force_numeric else None)
        kw = dict(self._pipe_kw)
        kw.setdefault("max_candidates", None)
        if self._X_val is not None:
            kw['X_val'] = self._X_val[:, cols]
        return Pipeline(
            self._X[:, cols], self._y, names,
            feature_types=ft, **kw,
        )

    def capture(self, fs, pipe):
        """Wrap a FeatureSet + pipe into a StageResult.

        Parameters
        ----------
        fs : FeatureSet
            Output of the last pipe.select() / pipe.raw() / etc.
        pipe : SoupPipe
            The pipe that produced fs.

        Returns
        -------
        StageResult
        """
        if not pipe.stages or pipe.stages[-1].output is not fs:
            raise ValueError("capture requires the final pipeline output for replay")
        return StageResult(fs, pipe, self._active, self._all_names)

    def subtract(self, stage_result):
        """Return a new pool with consumed base vars removed.

        Parameters
        ----------
        stage_result : StageResult
            Pass result whose ``.used`` set will be excluded.

        Returns
        -------
        FeaturePool
        """
        new = copy(self)
        new._active = [i for i in self._active
                       if self._all_names[i] not in stage_result.used]
        return new


# ---------------------------------------------------------------------------
# MultiPassResult — combined output of all passes
# ---------------------------------------------------------------------------

@dataclass
class MultiPassResult:
    """Combined result of a multi-pass soup pipeline.

    Attributes
    ----------
    X : ndarray (n, d_selected)
        Transformed feature matrix (all passes concatenated).
    names : list of str
        Selected feature names across all passes.
    pass_names : list of list of str
        Feature names per pass.
    used_base_vars : set of str
        Raw descriptor names consumed across all passes.
    unused_cols : list of int
        Column indices of descriptors never consumed.
    unused_names : list of str
        Names of descriptors never consumed.
    pipes : list of SoupPipe
        Fitted pipelines (one per pass).
    pass_cols : list of list of int
        Column indices used by each pass.
    """
    X: np.ndarray
    names: list
    pass_names: list
    used_base_vars: set
    unused_cols: list
    unused_names: list
    pipes: list
    pass_cols: list
    _all_names: list = field(repr=False)

    @classmethod
    def from_passes(cls, passes, all_names):
        """Build from a list of StageResults.

        Parameters
        ----------
        passes : list of StageResult
        all_names : list of str
            Full original feature name list.
        """
        if not passes:
            raise ValueError("No passes completed; check k and the minimum input columns")
        all_names = list(all_names)
        used = set()
        for sr in passes:
            used |= sr.used

        X = (np.hstack([sr.X for sr in passes])
             if len(passes) > 1 else passes[0].X)
        names = []
        pass_names = []
        for sr in passes:
            names.extend(sr.names)
            pass_names.append(sr.names)

        unused_cols = [i for i, n in enumerate(all_names) if n not in used]
        unused_names = [all_names[i] for i in unused_cols]

        return cls(
            X=X, names=names, pass_names=pass_names,
            used_base_vars=used,
            unused_cols=unused_cols, unused_names=unused_names,
            pipes=[sr.pipe for sr in passes],
            pass_cols=[sr.cols for sr in passes],
            _all_names=all_names,
        )

    def transform(self, X_new):
        """Transform new data through all passes."""
        parts = [pipe.transform(X_new[:, cols])
                 for pipe, cols in zip(self.pipes, self.pass_cols)]
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def transform_with_unused(self, X_new):
        """Transform + append unused raw descriptor columns."""
        X_soup = self.transform(X_new)
        if self.unused_cols:
            return np.hstack([X_soup, X_new[:, self.unused_cols]])
        return X_soup

    @property
    def n_soup_features(self):
        return self.X.shape[1]

    @property
    def n_unused(self):
        return len(self.unused_cols)


# ---------------------------------------------------------------------------
# Convenience wrapper (preserves old call-site compatibility)
# ---------------------------------------------------------------------------

def multi_pass(X, y, names, recipe, k=3, *, min_cols=3,
               force_numeric=True, **pipe_kw):
    """Run a soup recipe K times with auto-exclusion of consumed vars.

    Parameters
    ----------
    X : ndarray (n, d)
        Raw feature matrix.
    y : ndarray (n,)
        Target vector.
    names : list of str
        Feature names, length d.
    recipe : callable(SoupPipe) -> FeatureSet
        Stage recipe — receives a fresh SoupPipe, returns a FeatureSet.
    k : int
        Number of passes (default 3).
    min_cols : int
        Stop early if fewer columns remain (default 3).
    force_numeric : bool
        Force all features to numeric (default True).
    **pipe_kw
        Forwarded to SoupPipe (config, exclude_families, fold_eval_count, ...).

    Returns
    -------
    MultiPassResult
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive number of passes")
    pool = FeaturePool(X, y, names, min_cols=min_cols,
                       force_numeric=force_numeric, **pipe_kw)
    passes = []
    for _ in range(k):
        if pool.exhausted:
            break
        p = pool.build()
        fs = recipe(p)
        sr = pool.capture(fs, p)
        passes.append(sr)
        pool = pool.subtract(sr)

    return MultiPassResult.from_passes(passes, names)


# ---------------------------------------------------------------------------
# Recipe factories
# ---------------------------------------------------------------------------

def recipe_light(select=15):
    """Raw features only — no modulation. Fast baseline."""
    def r(p):
        return p.raw(select=select)
    return r


def recipe_std(select=15, raw_k=15, car_k=15, fm_k=15):
    """Raw + carriers + FM self-modulation."""
    def r(p):
        raw = p.raw(select=raw_k)
        car = p.carriers(raw, select=car_k)
        fm = p.fm(raw, raw, select=fm_k)
        return p.select(p.fuse(raw, car, fm), select=select)
    return r


def recipe_am(select=15, raw_k=15, am_k=15):
    """Raw + AM pairwise products/ratios."""
    def r(p):
        raw = p.raw(select=raw_k)
        am = p.am(raw, select=am_k)
        return p.select(p.fuse(raw, am), select=select)
    return r


def recipe_deep(select=15, raw_k=15, car_k=15, fm_k=15):
    """Raw + carriers + FM(carriers, raw) — deeper modulation chain."""
    def r(p):
        raw = p.raw(select=raw_k)
        car = p.carriers(raw, select=car_k)
        fm = p.fm(car, raw, select=fm_k)
        return p.select(p.fuse(raw, car, fm), select=select)
    return r


def recipe_carrier_first(select=20, car_k=15, am_k=15, fm_k=15, fm_thin=8):
    """Carriers first → AM + FM on carrier outputs → fuse → select.

    Builds carriers directly from all raw transforms, then uses AM for
    algebraic interactions (carrier × carrier) and FM for modulated
    oscillations (carrier warped by carrier).  ``fm_thin`` controls how
    many top carriers are fed to FM to stay within memory budget.
    """
    def r(p):
        raw_all = p.raw(select=None)
        car = p.carriers(raw_all, select=car_k)
        am = p.am(car, select=am_k)
        car_thin = p.select(car, select=fm_thin)
        fm = p.fm(car_thin, car_thin, select=fm_k)
        return p.select(p.fuse(car, am, fm), select=select)
    return r


def recipe_explore(select=15, raw_k=15, am_k=10, car_k=10, fm_k=10):
    """All main stages: raw + AM + carriers + deep FM → fuse → select."""
    def r(p):
        raw = p.raw(select=raw_k)
        am = p.am(raw, select=am_k)
        car = p.carriers(raw, select=car_k)
        fm = p.fm(car, raw, select=fm_k)
        return p.select(p.fuse(raw, am, car, fm), select=select)
    return r


def recipe_explore_deep(select=15, raw_k=15, am_k=10, car_k=10, fm_k=10,
                        deepen_depth=1, deepen_carry=10):
    """Wide parallel stages + recursive deepening on winners."""
    def r(p):
        raw = p.raw(select=raw_k)
        am = p.am(raw, select=am_k)
        car = p.carriers(raw, select=car_k)
        fm = p.fm(car, raw, select=fm_k)
        base = p.select(p.fuse(raw, am, car, fm), select=select)
        return p.deepen(base, select=select, max_depth=deepen_depth,
                        n_carry=deepen_carry)
    return r


def recipe_tower(raw_k=5, car_k=3, fm_k=1):
    """Single deep feature: raw → carriers → FM(carriers, raw).

    Multi-pass with k=10-15 gives diverse deep basis functions.
    """
    def r(p):
        raw = p.raw(select=raw_k)
        car = p.carriers(raw, select=car_k)
        return p.fm(car, raw, select=fm_k)
    return r


def recipe_tower_wm(raw_k=5, car_k=3, fm_k=3, wm_k=1):
    """Deepest: raw → carriers → FM → window modulation."""
    def r(p):
        raw = p.raw(select=raw_k)
        car = p.carriers(raw, select=car_k)
        fm = p.fm(car, raw, select=fm_k)
        return p.wm(fm, raw, select=wm_k)
    return r
