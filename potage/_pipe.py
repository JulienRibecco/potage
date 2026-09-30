"""Pipeline -- Modular pipeline for composable feature engineering.

Unlike StagedSoup (fixed linear pipeline), Pipeline lets users route
different feature subsets through different stages, control per-step
selection budgets, and compose custom DAGs via feature set handles.

Example::

    pipe = Pipeline(X, y, names)
    s0 = pipe.raw(select=20)
    s1 = pipe.am(s0, select=30)
    s2 = pipe.carriers(s0, select=15)
    merged = pipe.fuse(s1, s2)
    final = pipe.am(merged, select=50)
"""

import time
from copy import deepcopy
import warnings

import numpy as np

from . import _workflows
from ._workflows import _thin_grid
from ._library import PrimitiveLibrary
from ._dispatch import select_features, validate_selection
from ._provenance import input_names
from ._state import StateMixin, atomic_stage
from ._records import FeatureSet, Stage
from ._replay import ReplayMixin
from ._budget import (
    BudgetMixin,
    CandidateBudgetExceeded,
    MemoryBudgetExceeded,
    _DEFAULT_MAX_CANDIDATES,
    _SENTINEL_AUTO,
    _resolve_max_memory,
    _estimate_step_peak_bytes,
)
from ._config import SoupConfig
from ._algebra import build_stage0_raw, build_stage1_am
from ._carriers import build_stage2_static_carriers, _build_targeted_carriers
from ._frequency import fft_freqs, calibrate_carrier_freqs
from ._feature_utils import _infer_feature_types, _derive_masks
from ._modulation import (
    _fm_sin_kernel,
    _pm_sin_kernel,
    _fm_tri_kernel,
    _fm_pulse_kernel,
    _wm_center_kernel,
    _wm_width_kernel,
    _apply_modulation,
)
from ._fields import build_stage_fields, neuron_probe_groups, build_probe_zones
from ._kfold import _align_val_cols
from ._cross import _apply_cross_modulation, _build_cross_am


class Pipeline(StateMixin, ReplayMixin, BudgetMixin):
    """Modular pipeline builder for soup feature engineering.

    Each method generates features from one or more FeatureSet inputs,
    runs selection, and returns a new FeatureSet handle.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Raw input data.
    y : ndarray of shape (n_samples,)
        Target variable.
    names : list of str
        Feature names for the input columns.
    feature_types : array-like of str, optional
        Per-column type: 'numeric', 'bool', or 'categorical'.
        Inferred from data if not provided.
    config : SoupConfig, optional
        Configuration for frequency grids and modulation depths.
    max_memory : int, float, str, or None
        Memory budget per step.  ``'auto'`` (default) uses 50% of
        detected available RAM.  A ``float`` in (0, 1] is treated as
        a fraction of available RAM.  An ``int`` is an explicit byte
        count.  ``None`` disables the memory guard entirely.
    X_val : ndarray of shape (n_val, n_features), optional
        Validation input data. When provided with y_val, selection
        scores on held-out data and val features propagate through
        every pipeline step.
    y_val : ndarray of shape (n_val,), optional
        Validation target. Required when X_val is provided.
    fold_eval_count : int or None
        Number of cross-validation folds for selection scoring.
        Greedy selection ranks candidates by validation R². OMP ranks by
        training-residual correlation and reports validation R². Correlation
        selection averages relevance/redundancy scores, not R². Defaults to 3.
        These are adaptive selection diagnostics on precomputed features,
        not independent generalization estimates. Keep an outer holdout.
        Set to ``None`` to disable fold scoring.
        Ignored when explicit X_val/y_val is provided.
    fold_seed : int
        Random seed for creating the K folds. Default 42.
    """

    _DEFAULT_FOLD_EVAL = 3

    def __init__(self, X, y, names, feature_types=None, config=None,
                 max_candidates=_DEFAULT_MAX_CANDIDATES,
                 max_memory=_SENTINEL_AUTO,
                 X_val=None, y_val=None,
                 fold_eval_count=_DEFAULT_FOLD_EVAL, fold_seed=42,
                 task='regression',
                 y_base=None, y_val_base=None,
                 exclude_families=None):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.names = input_names(names)
        self.config = deepcopy(config) if config is not None else SoupConfig()
        if self.X.ndim != 2 or self.y.shape != (self.X.shape[0],):
            raise ValueError("X must be a matrix and y must have one value per row")
        if len(self.names) != self.X.shape[1] or len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique and match the input columns")
        if fold_eval_count is not None and (not isinstance(fold_eval_count, int) or fold_eval_count < 2):
            raise ValueError("fold_eval_count must be None or an integer >= 2")
        self.max_candidates = max_candidates
        self.max_memory = _resolve_max_memory(max_memory)
        self._candidates_used = 0
        self._memory_high_water = 0

        # Task type
        if task not in ('regression', 'classification'):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
        self.task = task

        # Validation split (X_val may be provided without y_val for
        # transform-only usage — selection won't score on val in that case)
        if y_val is not None and X_val is None:
            raise ValueError("y_val requires X_val")
        if X_val is not None:
            self.X_val = np.asarray(X_val, dtype=np.float64)
            self.y_val = (np.asarray(y_val, dtype=np.float64)
                          if y_val is not None else None)
            if self.X_val.ndim != 2:
                raise ValueError("X_val must be a matrix")
            if self.y_val is not None and self.y_val.shape != (len(self.X_val),):
                raise ValueError("y_val must have one value per validation row")
            if self.X_val.shape[1] != self.X.shape[1]:
                raise ValueError(
                    f"X_val has {self.X_val.shape[1]} columns, "
                    f"expected {self.X.shape[1]}"
                )
        else:
            self.X_val = None
            self.y_val = None

        # Base model predictions — selection targets residuals y - y_base,
        # R² is reported relative to original y so it starts at base_r2.
        if y_base is not None:
            self._y_base = np.asarray(y_base, dtype=np.float64)
            self._y_target = self.y - self._y_base
            y_var = np.var(self.y)
            self._base_r2 = 1.0 - np.var(self._y_target) / y_var if y_var > 0 else 0.0
        else:
            self._y_base = None
            self._y_target = self.y
            self._base_r2 = 0.0

        if y_val_base is not None and self.y_val is not None:
            self._y_val_target = self.y_val - np.asarray(y_val_base, dtype=np.float64)
        else:
            self._y_val_target = self.y_val

        # K-fold evaluation (alternative to explicit val)
        self.fold_eval_count = fold_eval_count
        self.fold_seed = fold_seed

        self._exclude_families = set(exclude_families) if exclude_families else set()

        self._stages = []
        self._checkpoints = []
        self._fs_to_stage = {}  # id(FeatureSet) → stage index
        self._candidate_families_seen = set()  # all families offered to selection

        if feature_types is not None:
            self.feature_types = np.asarray(feature_types, dtype='<U11')
        else:
            self.feature_types = _infer_feature_types(
                self.X, names=self.names)

        if self.feature_types.shape != (self.X.shape[1],) or not np.isin(
                self.feature_types, ['numeric', 'bool', 'categorical']).all():
            raise ValueError("feature_types must contain one numeric/bool/categorical type per column")

        # FFT grid calibration: replace 'fft' with data-driven frequencies
        if self.config.carrier_freqs == 'fft':
            # Set a temporary default to prevent recursion during calibration
            self.config.carrier_freqs = [0.5, 1.0, 2.0, 3.0]
            self.config.carrier_freqs = calibrate_carrier_freqs(
                self.X, self.y, self.names, self.config,
                n_freqs=self.config.calibrate_n_freqs,
                exclude_families=self._exclude_families,
            )

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Object ids change across processes; rebuild handles after unpickling.
        self._fs_to_stage = {id(stage.output): i
                             for i, stage in enumerate(self._stages)}

    def _guard_candidates(self, step, *inputs):
        """Check cumulative budget and memory budget, then track usage.

        Raises CandidateBudgetExceeded if adding this step's candidates
        would push the pipeline total over max_candidates.
        Raises MemoryBudgetExceeded if the step's peak allocation would
        exceed max_memory.
        Always tracks the count, even when max_candidates is None.
        """
        est = self.predict_candidates(step, *inputs)
        step_count = est['candidates']
        if (self.max_candidates is not None
                and self._candidates_used + step_count > self.max_candidates):
            raise CandidateBudgetExceeded(
                step, step_count, self.max_candidates,
                self._candidates_used, est['n_samples'], est['breakdown'],
            )
        # Memory guard: estimate peak bytes including dedup + kfold overhead
        if self.max_memory is not None:
            n_total = est['n_samples']
            if self.X_val is not None:
                n_total += self.X_val.shape[0]
            kfold_k = (self.fold_eval_count
                        if self.fold_eval_count and self.fold_eval_count > 1
                        and self.y_val is None
                        else None)
            mem = _estimate_step_peak_bytes(step_count, n_total, kfold_k)
            if mem['peak_bytes'] > self.max_memory:
                raise MemoryBudgetExceeded(
                    step, mem['peak_bytes'], self.max_memory,
                    est['n_samples'],
                    breakdown_bytes={
                        'candidates': mem['candidate_bytes'],
                        'dedup': mem['dedup_bytes'],
                        'selection': mem['selection_bytes'],
                        'kfold': mem['kfold_bytes'],
                    },
                )
            self._memory_high_water = max(self._memory_high_water,
                                           mem['peak_bytes'])
        self._candidates_used += step_count

    def _record(self, name, fs, history, n_gen, n_filt,
                select_budget, method, elapsed, input_dims,
                parent_ids=(), step_type='raw', per_class_histories=None,
                surviving_names=None, metadata=None):
        """Create a Stage and append to self._stages."""
        stage_idx = len(self._stages)
        self._stages.append(Stage(
            name=name, output=fs, history=history,
            candidates_generated=n_gen, candidates_after_filter=n_filt,
            select_budget=select_budget, method=method,
            elapsed=elapsed, input_dims=input_dims,
            parent_ids=parent_ids, step_type=step_type,
            per_class_histories=per_class_histories,
            surviving_names=surviving_names, metadata=metadata,
        ))
        self._fs_to_stage[id(fs)] = stage_idx
        self._checkpoints.append(self.checkpoint())

    @property
    def stages(self):
        """Shallow copy of the pipeline's Stage records."""
        return list(self._stages)

    def recap(self):
        """Formatted summary table of all pipeline stages.

        Returns
        -------
        str
        """
        if not self._stages:
            return 'Pipeline: no stages recorded'

        n_samples = self.X.shape[0]
        has_base = self._base_r2 > 0
        header = f'Pipeline: {len(self._stages)} stages, {n_samples:,} samples'
        if has_base:
            header += f' (base R\u00b2={self._base_r2:.4f})'
        sep = '\u2500' * 72
        r2_label = 'R\u00b2(total)' if has_base else 'R\u00b2'
        col_header = (f'  {"#":>2}  {"Step":<20s} {"Dims":>10s}'
                      f'  {"Candidates":>14s}  {r2_label:>10s}'
                      f'  {"Method":<6s}  {"Time":>5s}')

        lines = [header, sep, col_header, sep]

        total_gen = 0
        for i, s in enumerate(self._stages, 1):
            total_gen += s.candidates_generated

            # Dims column: input→output
            dims = f'{s.input_dims}\u2192{s.n_selected}'

            # Candidates column
            if s.history is not None:
                cands = f'{s.candidates_generated:,}\u2192{s.candidates_after_filter:,}'
            else:
                cands = '\u2014'

            # R² column — when base model is set, show total R²
            if s.final_r2 is not None:
                if has_base:
                    total_r2 = self._base_r2 + (1 - self._base_r2) * s.final_r2
                    r2 = f'{total_r2:.4f}'
                else:
                    r2 = f'{s.final_r2:.4f}'
            else:
                r2 = '\u2014'

            # Method column
            method_str = s.method or '\u2014'

            # Time column
            t = f'{s.elapsed:.1f}s'

            lines.append(
                f'  {i:>2}  {s.name:<20s} {dims:>10s}'
                f'  {cands:>14s}  {r2:>10s}'
                f'  {method_str:<6s}  {t:>5s}'
            )

        lines.append(sep)

        # Budget line
        if self.max_candidates is not None:
            pct = self._candidates_used / self.max_candidates * 100
            lines.append(
                f'Budget (estimated): {self._candidates_used:,} / {self.max_candidates:,}'
                f' ({pct:.1f}%)'
            )

        # Memory line
        if self.max_memory is not None and self._memory_high_water > 0:
            peak_mb = self._memory_high_water / 2**20
            max_mb = self.max_memory / 2**20
            mem_pct = self._memory_high_water / self.max_memory * 100
            lines.append(
                f'Memory: peak ~{peak_mb:.0f} MB / {max_mb:.0f} MB'
                f' ({mem_pct:.1f}%)'
            )

        # Unused families advisory
        unused = self.unused_families()
        if unused:
            lines.append(
                f'Unused families: {", ".join(sorted(unused))}'
                f'  (never selected — consider excluding to free budget)'
            )

        return '\n'.join(lines)

    def unused_families(self):
        """Families that were generated as candidates but never selected.

        Returns
        -------
        set of str
            Family names that survived filter/dedup but were never picked
            by any selection step.
        """
        selected = set()
        for stage in self._stages:
            if stage.history is not None and len(stage.history) > 0:
                selected.update(stage.history.selected_families())
        return self._candidate_families_seen - selected

    @atomic_stage
    def raw(self, select, method='omp'):
        """Stage 0: raw & unary transforms on input data.

        Parameters
        ----------
        select : int or None
            Number of features to select.  None returns all surviving
            features after filter/dedup (no selection).
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('raw')
        is_numeric, _, _ = _derive_masks(self.feature_types)
        cols, pnames, fams = build_stage0_raw(
            self.X, self.names, is_numeric,
            feature_types=self.feature_types,
        )
        # Build val features using train stats (reference_X=self.X)
        cols_val = None
        if self.X_val is not None:
            cols_val_list, vnames, _ = build_stage0_raw(
                self.X_val, self.names, is_numeric,
                reference_X=self.X,
                feature_types=self.feature_types,
            )
            # Align val columns to train by name (one-hot may differ)
            cols_val = _align_val_cols(pnames, vnames, cols_val_list,
                                       self.X_val.shape[0])
        # Preserve original types for raw passthrough and one-hot
        input_type = dict(zip(self.names, self.feature_types))
        ctypes = []
        for name, fam in zip(pnames, fams):
            if fam == 'raw' and name in input_type:
                ctypes.append(str(input_type[name]))
            elif fam == 'onehot':
                ctypes.append('bool')
            else:
                ctypes.append('numeric')
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method,
            candidate_types=ctypes, cols_val=cols_val)
        self._record('raw', fs, history, n_gen, n_filt,
                     select, method, time.time() - t0, self.X.shape[1],
                     parent_ids=(), step_type='raw',
                     per_class_histories=per_class,
                     surviving_names=surv)
        return fs

    @atomic_stage
    def am(self, *sets, select, method='omp'):
        """Amplitude modulation: pairwise binary ops.

        With one FeatureSet, generates self-pairs (within the set).
        With multiple FeatureSets, generates cross-pairs only (between
        distinct sets, no intra-set pairs).

        Parameters
        ----------
        *sets : FeatureSet
            One or more input feature sets.
        select : int
            Number of features to select.
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.

        Returns
        -------
        FeatureSet
        """
        if len(sets) == 0:
            raise ValueError("am requires at least 1 FeatureSet")
        t0 = time.time()
        self._guard_candidates('am', *sets)
        if len(sets) == 1:
            s = sets[0]
            is_numeric, _, _ = _derive_masks(s.feature_types)
            cols, pnames, fams = build_stage1_am(
                s.X, s.names, is_numeric,
                config=self.config,
                feature_types=s.feature_types,
            )
            # Reuse training statistics for ratios and power exponents.
            cols_val = None
            if s._X_val is not None:
                is_num_v, _, _ = _derive_masks(s.feature_types)
                cols_val_list, vnames, _ = build_stage1_am(
                    s._X_val, s.names, is_num_v,
                    config=self.config,
                    feature_types=s.feature_types, reference_X=s.X,
                )
                # Align val columns to train by name (conditional transforms
                # like log/reciprocal may produce different valid columns)
                cols_val = _align_val_cols(pnames, vnames, cols_val_list,
                                           s._X_val.shape[0])
        else:
            cols, pnames, fams = _build_cross_am(sets, self.config)
            cols_val = None
            has_val = all(s._X_val is not None for s in sets)
            if has_val:
                # Compute train stds for deterministic filtering decisions
                train_X = np.hstack([s.X for s in sets])
                train_stds = train_X.std(axis=0)
                val_sets = [FeatureSet(s._X_val, s.names, s.feature_types)
                            for s in sets]
                cols_val, _, _ = _build_cross_am(val_sets, self.config,
                                                 ref_stds=train_stds,
                                                 reference_X=train_X)
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        # Build input description
        input_dims = sum(s.X.shape[1] for s in sets)
        dim_parts = [f'{s.X.shape[1]}d' for s in sets]
        input_desc = '+'.join(dim_parts)
        parent_ids = tuple(self._fs_to_stage[id(s)] for s in sets)
        step_type = 'am_single' if len(sets) == 1 else 'am_cross'
        self._record(f'am({input_desc})', fs, history, n_gen, n_filt,
                     select, method, time.time() - t0, input_dims,
                     parent_ids=parent_ids, step_type=step_type,
                     per_class_histories=per_class,
                     surviving_names=surv)
        return fs

    @atomic_stage
    def carriers(self, s, select, method='omp', feature_space='linear'):
        """Stage 2: static carrier waveforms on a FeatureSet.

        Parameters
        ----------
        s : FeatureSet
            Input feature set (numeric features only).
        select : int
            Number of features to select.
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.
        feature_space : str, default 'linear'
            Coordinate space applied before [0,1] normalisation:
            'linear'  — no transform (backward-compatible default).
            'log'     — signed natural log: sign(x)*log(1+|x|).
            'log2'    — signed base-2 log: sign(x)*log2(1+|x|).
            'log10'   — signed base-10 log. For pH/dB/magnitude scales.
            'sqrt'    — signed square root: sign(x)*sqrt(|x|).
            'cbrt'    — signed cube root: sign(x)*|x|^(1/3). Weyl/shell space.
            'exp'     — signed exp: sign(x)*expm1(|x|).
            'arcsinh' — arcsinh(x). Analytic at 0, best log-like for general use.
            'logit'   — logit(x). For proportion/probability features in (0,1).
            'tanh'    — tanh(x). Symmetric compression of heavy tails.
            'pow:<p>' — sign(x)*|x|^p for arbitrary p, e.g. 'pow:0.33'.
            Step carriers use raw percentile thresholds and are unaffected.
            Feature names are annotated: sin(1.0π*log2(fw_mf_ratio)).

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('carriers', s)
        is_numeric, _, _ = _derive_masks(s.feature_types)
        cols, pnames, fams = build_stage2_static_carriers(
            s.X, s.names, is_numeric, self.config,
            feature_space=feature_space,
        )
        # Build val carriers using train stats (reference_X=s.X)
        cols_val = None
        if s._X_val is not None:
            cols_val_list, _, _ = build_stage2_static_carriers(
                s._X_val, s.names, is_numeric, self.config,
                reference_X=s.X,
                feature_space=feature_space,
            )
            cols_val = cols_val_list
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        parent_ids = (self._fs_to_stage[id(s)],)
        meta = {'feature_space': feature_space} if feature_space != 'linear' else None
        self._record(f'carriers({s.X.shape[1]}d)', fs, history, n_gen, n_filt,
                     select, method, time.time() - t0, s.X.shape[1],
                     parent_ids=parent_ids, step_type='carriers',
                     per_class_histories=per_class,
                     surviving_names=surv, metadata=meta)
        return fs

    def discover_carrier_space(self, fs_raw, select=15, method='omp',
                               spaces=None, include_raw=True):
        """Sweep carrier spaces and fuse, returning the best combined set.

        Runs ``carriers()`` for each space in *spaces*, fuses all results
        with the raw features (if *include_raw*), and selects the top
        *select* features globally.  Reports which space each winning
        feature came from so you can see what the data prefers.

        Parameters
        ----------
        fs_raw : FeatureSet
            Raw feature set (output of ``pipe.raw(...)``).
        select : int
            Final number of features to select across all spaces.
        method : str
            OMP/greedy/correlation selection.
        spaces : list of str or None
            Feature spaces to try.  Defaults to
            ``['linear', 'arcsinh', 'log', 'sqrt', 'cbrt', 'tanh']`` —
            a practical sweep covering the most distinct regimes.
        include_raw : bool
            Whether to include the raw features in the final pool.

        Returns
        -------
        FeatureSet
            Selected features with names annotated by space.
        dict
            ``{'winner_counts': {space: n_features_selected}, 'spaces': spaces}``
            — how many features came from each space.
        """
        return _workflows.discover_carrier_space(
            self, fs_raw=fs_raw,
            select=select,
            method=method,
            spaces=spaces,
            include_raw=include_raw)

    @atomic_stage
    def fields(self, s, select, method='omp'):
        """Stage 7: spatial field features on numeric column pairs.

        For each pair of numeric columns in *s*, builds a 2D percentile grid
        and computes Gaussian/inverse/nearest (Euclidean) and zone/ezone
        (Mahalanobis) field features.

        Parameters
        ----------
        s : FeatureSet
            Input feature set.
        select : int
            Number of features to select.
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('fields', s)
        is_numeric, _, _ = _derive_masks(s.feature_types)
        cols, pnames, fams = build_stage_fields(
            s.X, s.names, is_numeric, self.config,
        )
        cols_val = None
        if s._X_val is not None:
            cols_val_list, _, _ = build_stage_fields(
                s._X_val, s.names, is_numeric, self.config,
                reference_X=s.X,
            )
            cols_val = cols_val_list
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        parent_ids = (self._fs_to_stage[id(s)],)
        self._record(f'fields({s.X.shape[1]}d)', fs, history, n_gen, n_filt,
                     select, method, time.time() - t0, s.X.shape[1],
                     parent_ids=parent_ids, step_type='fields',
                     per_class_histories=per_class,
                     surviving_names=surv)
        return fs

    @atomic_stage
    def probe_fields(self, s, select, method='omp',
                     n_neurons=5, bind_k=1, n_seeds=5,
                     max_per_group=10, n_epochs=2000):
        """Neuron-probe-guided spatial zone features.

        Fits a tiny NN to discover feature groupings, then builds zone
        features only within each neuron-defined group.  This reduces
        the combinatorial cost from C(d,2) to sum(C(g_i,2)) where each
        g_i is a small neuron group.

        Parameters
        ----------
        s : FeatureSet
            Input feature set.
        select : int
            Number of features to select.
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.
        n_neurons : int
            Number of hidden neurons in the probe NN.
        bind_k : int
            Each feature binds to its top-k neurons (1=disjoint groups).
        n_seeds : int
            Number of random seeds to average probe weights over.
        max_per_group : int
            Cap features per neuron group.
        n_epochs : int
            Training epochs for each probe NN.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('probe_fields', s)
        is_numeric, _, _ = _derive_masks(s.feature_types)

        # Phase 1: neuron probe to discover feature groups
        groups, W_avg = neuron_probe_groups(
            s.X, self._y_target, n_neurons=n_neurons, bind_k=bind_k,
            n_seeds=n_seeds, max_per_group=max_per_group,
            n_epochs=n_epochs)

        if not groups:
            warnings.warn("probe_fields: no groups with >=2 features found")
            # Return empty feature set
            empty_X = np.empty((s.X.shape[0], 0))
            empty_fs = FeatureSet(empty_X, [], np.array([], dtype='<U11'))
            parent_ids = (self._fs_to_stage[id(s)],)
            self._record(f'probe_fields({s.X.shape[1]}d,{n_neurons}n)', empty_fs,
                         None, 0, 0, select, method, time.time() - t0,
                         s.X.shape[1], parent_ids=parent_ids,
                         step_type='probe_fields')
            return empty_fs

        # Phase 2: build zones within probe groups
        cols, pnames, fams = build_probe_zones(
            s.X, s.names, groups, config=self.config)
        cols_val = None
        if s._X_val is not None:
            cols_val_list, _, _ = build_probe_zones(
                s._X_val, s.names, groups, config=self.config,
                reference_X=s.X)
            cols_val = cols_val_list

        # Phase 3: select
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        parent_ids = (self._fs_to_stage[id(s)],)

        # Store probe groups in metadata for transform replay
        # Only store indices (not weights) — that's all replay needs
        serializable_groups = [{'indices': g['indices']} for g in groups]
        metadata = {'probe_groups': serializable_groups}

        self._record(f'probe_fields({s.X.shape[1]}d,{n_neurons}n)', fs,
                     history, n_gen, n_filt,
                     select, method, time.time() - t0, s.X.shape[1],
                     parent_ids=parent_ids, step_type='probe_fields',
                     per_class_histories=per_class,
                     surviving_names=surv, metadata=metadata)
        return fs

    @atomic_stage
    def fft_carriers(self, s, select, method='omp', n_per_feature=5,
                     min_power_ratio=0.1, max_k=None,
                     y_target=None):
        """Stage 2 (FFT-guided): detect carrier frequencies from data.

        Instead of sweeping a fixed frequency grid, FFTs y-vs-sorted-feature
        for each input column to find the actual periodicities in the
        y-feature relationship.  Then generates sin/cos/tri/step/pulse/gauss
        carriers at the detected frequencies only.

        This is the data-driven counterpart of carriers().  The paper
        "When Periodic Primitives Outperform Neural Networks" (Ribecco 2026)
        proves that explicit frequency identification (FFT) + closed-form
        estimation (Ridge) beats learned representations.  This method
        applies the same principle to tabular feature engineering.

        Parameters
        ----------
        s : FeatureSet
            Input feature set (numeric features only).
        select : int
            Number of features to select.
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.
        n_per_feature : int
            Maximum number of FFT-detected frequencies per feature.
        min_power_ratio : float
            Minimum spectral power as fraction of strongest peak.
        max_k : float or None
            Maximum carrier k value.  None = Nyquist-aware auto.
        y_target : (n,) array or None
            Target for FFT frequency detection.  If None, uses the
            pipe's y.  Pass ridge residuals to find periodic structure
            that linear features can't capture — frequencies in the
            residual are exactly what carriers need to explain.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('fft_carriers', s)
        is_numeric, _, _ = _derive_masks(s.feature_types)
        num_idx = np.where(is_numeric)[0]

        # Use custom y for FFT detection (e.g. residuals), but selection
        # still uses the pipe's actual target
        y_fft = y_target if y_target is not None else self.y

        # Detect per-feature frequencies via FFT
        periodicities = fft_freqs(
            s.X, y_fft, num_idx,
            n_per_feature=n_per_feature,
            phase_scale=self.config.carrier_phase_scale,
            min_power_ratio=min_power_ratio,
            max_k=max_k,
        )

        n_features_with_signal = len(periodicities)
        n_total_freqs = sum(len(v) for v in periodicities.values())

        if n_features_with_signal == 0:
            # No periodic structure found — fall back to static carriers
            warnings.warn(
                "fft_carriers: no periodic structure detected in any feature; "
                "falling back to static carriers")
            return self.carriers(s, select, method)

        # Build carriers at detected frequencies
        cols, pnames, fams = _build_targeted_carriers(
            s.X, s.names, is_numeric, self.config, periodicities,
        )

        # Val carriers using train stats
        cols_val = None
        if s._X_val is not None:
            cols_val_list, _, _ = _build_targeted_carriers(
                s._X_val, s.names, is_numeric, self.config, periodicities,
                reference_X=s.X,
            )
            cols_val = cols_val_list

        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        parent_ids = (self._fs_to_stage[id(s)],)
        self._record(
            f'fft_carriers({s.X.shape[1]}d, '
            f'{n_features_with_signal}feat, {n_total_freqs}freqs)',
            fs, history, n_gen, n_filt,
            select, method, time.time() - t0, s.X.shape[1],
            parent_ids=parent_ids, step_type='fft_carriers',
            per_class_histories=per_class,
            surviving_names=surv)

        # Store detected periodicities for transform replay
        stage_id = self._fs_to_stage[id(fs)]
        stage_obj = self._stages[stage_id]
        if stage_obj._metadata is None:
            stage_obj._metadata = {}
        stage_obj._metadata['fft_periodicities'] = periodicities

        return fs

    @atomic_stage
    def fm(self, carrier, mod, select, method='omp'):
        """FM/PM modulation: carrier features modulated by mod features.

        Parameters
        ----------
        carrier : FeatureSet
            Carrier feature set.
        mod : FeatureSet
            Modulator feature set.
        select : int
            Number of features to select.
        method : str
            Selection method: 'omp', 'greedy', or 'correlation'.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('fm', carrier, mod)

        fm_grid = [{'k': k, 'd': d}
                   for k in self.config.carrier_freqs
                   for d in self.config.fm_depths]
        pm_grid = [{'k': k, 'p': p}
                   for k in self.config.carrier_freqs
                   for p in self.config.pm_depths]

        kernels = [
            (_fm_sin_kernel, fm_grid, 'fm_sin'),
            (_pm_sin_kernel, pm_grid, 'pm_sin'),
        ]

        cols, pnames, fams = _apply_cross_modulation(
            carrier.X, carrier.names, carrier.feature_types,
            mod.X, mod.names, mod.feature_types,
            self.config, kernels,
        )
        # Build val modulation using train norm01 stats
        cols_val = None
        has_val = carrier._X_val is not None and mod._X_val is not None
        if has_val:
            cols_val, _, _ = _apply_cross_modulation(
                carrier._X_val, carrier.names, carrier.feature_types,
                mod._X_val, mod.names, mod.feature_types,
                self.config, kernels,
                ref_X_c=carrier.X, ref_X_m=mod.X,
            )
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        input_dims = carrier.X.shape[1] + mod.X.shape[1]
        parent_ids = (self._fs_to_stage[id(carrier)],
                      self._fs_to_stage[id(mod)])
        self._record(
            f'fm({carrier.X.shape[1]}d,{mod.X.shape[1]}d)',
            fs, history, n_gen, n_filt,
            select, method, time.time() - t0, input_dims,
            parent_ids=parent_ids, step_type='fm',
            per_class_histories=per_class,
            surviving_names=surv)
        return fs

    @atomic_stage
    def wm(self, carrier, mod, select, method='omp'):
        """Window modulation: Gaussian windows with center/width shifts.

        Parameters
        ----------
        carrier : FeatureSet
            Carrier feature set.
        mod : FeatureSet
            Modulator feature set.
        select : int or None
            Number of features to select.
        method : str
            Selection method.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('wm', carrier, mod)

        center_grid = [{'q': q} for q in self.config.wm_center_pcts]
        width_grid = [{'q': q, 'wmod': wmod}
                      for q in self.config.wm_center_pcts
                      for wmod in self.config.wm_width_mods]

        kernels = [
            (_wm_center_kernel, center_grid, 'wm_center'),
            (_wm_width_kernel, width_grid, 'wm_width'),
        ]

        cols, pnames, fams = _apply_cross_modulation(
            carrier.X, carrier.names, carrier.feature_types,
            mod.X, mod.names, mod.feature_types,
            self.config, kernels,
        )
        cols_val = None
        has_val = carrier._X_val is not None and mod._X_val is not None
        if has_val:
            cols_val, _, _ = _apply_cross_modulation(
                carrier._X_val, carrier.names, carrier.feature_types,
                mod._X_val, mod.names, mod.feature_types,
                self.config, kernels,
                ref_X_c=carrier.X, ref_X_m=mod.X,
            )
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        input_dims = carrier.X.shape[1] + mod.X.shape[1]
        parent_ids = (self._fs_to_stage[id(carrier)],
                      self._fs_to_stage[id(mod)])
        self._record(
            f'wm({carrier.X.shape[1]}d,{mod.X.shape[1]}d)',
            fs, history, n_gen, n_filt,
            select, method, time.time() - t0, input_dims,
            parent_ids=parent_ids, step_type='wm',
            per_class_histories=per_class,
            surviving_names=surv)
        return fs

    @atomic_stage
    def fm_nonsmooth(self, carrier, mod, select, method='omp'):
        """FM with non-smooth carriers (triangle, pulse).

        Parameters
        ----------
        carrier : FeatureSet
            Carrier feature set.
        mod : FeatureSet
            Modulator feature set.
        select : int or None
            Number of features to select.
        method : str
            Selection method.

        Returns
        -------
        FeatureSet
        """
        t0 = time.time()
        self._guard_candidates('fm_nonsmooth', carrier, mod)

        fm_tri_grid = [{'k': k, 'd': d}
                       for k in self.config.carrier_freqs
                       for d in self.config.fm_depths]
        fm_pulse_grid = [{'q': q, 'w': w}
                         for q in self.config.gauss_center_pcts
                         for w in self.config.gauss_widths]

        kernels = [
            (_fm_tri_kernel, fm_tri_grid, 'fm_tri'),
            (_fm_pulse_kernel, fm_pulse_grid, 'fm_pulse'),
        ]

        cols, pnames, fams = _apply_cross_modulation(
            carrier.X, carrier.names, carrier.feature_types,
            mod.X, mod.names, mod.feature_types,
            self.config, kernels,
        )
        cols_val = None
        has_val = carrier._X_val is not None and mod._X_val is not None
        if has_val:
            cols_val, _, _ = _apply_cross_modulation(
                carrier._X_val, carrier.names, carrier.feature_types,
                mod._X_val, mod.names, mod.feature_types,
                self.config, kernels,
                ref_X_c=carrier.X, ref_X_m=mod.X,
            )
        fs, history, n_gen, n_filt, per_class, surv = self._select_top(
            cols, pnames, fams, select, method, cols_val=cols_val)
        input_dims = carrier.X.shape[1] + mod.X.shape[1]
        parent_ids = (self._fs_to_stage[id(carrier)],
                      self._fs_to_stage[id(mod)])
        self._record(
            f'fm_nonsmooth({carrier.X.shape[1]}d,{mod.X.shape[1]}d)',
            fs, history, n_gen, n_filt,
            select, method, time.time() - t0, input_dims,
            parent_ids=parent_ids, step_type='fm_nonsmooth',
            per_class_histories=per_class,
            surviving_names=surv)
        return fs

    @atomic_stage
    def select(self, fs, select, method='omp'):
        """Select top features from an existing FeatureSet.

        Parameters
        ----------
        fs : FeatureSet
            Input feature set (already generated, no new candidates built).
        select : int
            Number of features to select.
        method : str
            Selection method.

        Returns
        -------
        FeatureSet
        """
        validate_selection(select, method)
        from ._library import SelectionHistory
        t0 = time.time()

        n_train = fs.X.shape[0]
        n_generated = fs.X.shape[1]
        if n_generated == 0 or select == 0:
            empty = FeatureSet(
                np.empty((n_train, 0)), [], np.array([], dtype='<U11'),
                _X_val=np.empty((fs._X_val.shape[0], 0)) if fs._X_val is not None else None,
            )
            parent_ids = (self._fs_to_stage[id(fs)],)
            self._record(f'select({n_generated}d)', empty, SelectionHistory(), 0, 0,
                         select, method, time.time() - t0, n_generated,
                         parent_ids=parent_ids, step_type='select')
            return empty

        families = fs._families if fs._families else ['unknown'] * n_generated
        lib = PrimitiveLibrary(fs.X, fs.names, families,
                               X_val=fs._X_val)
        n_select = min(select, lib.X.shape[1])

        has_val = fs._X_val is not None and self.y_val is not None
        y_val = self._y_val_target if has_val else None

        history, per_class_histories = select_features(
            lib, self._y_target, n_select, method, task=self.task,
            folds=self.fold_eval_count, seed=self.fold_seed,
            y_val=self._y_val_target if has_val else None)

        sel_names = history.selected_names()
        name_to_idx = {n: i for i, n in enumerate(lib.names)}
        sel_idx = [name_to_idx[n] for n in sel_names]
        X_sel = lib.X[:, sel_idx]
        X_val_sel = lib.X_val[:, sel_idx] if lib.X_val is not None else None
        ft = np.asarray(fs.feature_types)[sel_idx]
        sel_families = [lib.families[i] for i in sel_idx]
        result = FeatureSet(X_sel, sel_names, ft, _X_val=X_val_sel,
                            _families=sel_families)

        parent_ids = (self._fs_to_stage[id(fs)],)
        self._record(f'select({n_generated}d)', result, history,
                     n_generated, n_generated,
                     select, method, time.time() - t0, n_generated,
                     parent_ids=parent_ids, step_type='select',
                     per_class_histories=per_class_histories)
        return result

    def auto_run(self, select=20, max_stage=5, method='omp',
                 early_stop_marginal=0.003, patience=2, verbose=False):
        """Automated staged pipeline replicating StagedSoup.

        Builds features in stages of increasing complexity, fusing
        accumulated pools and selecting after each stage.

        Parameters
        ----------
        select : int
            Number of features to select at each checkpoint.
        max_stage : int
            Maximum stage (0=raw, 1=AM, 2=carriers, 3=FM, 4=FM_tri, 5=WM).
        method : str
            Selection method.
        early_stop_marginal : float
            Minimum marginal R² gain to continue.
        patience : int
            Consecutive stages below threshold before stopping.
        verbose : bool

        Returns
        -------
        FeatureSet
            Final selected features.
        """
        return _workflows.auto_run(
            self, select=select,
            max_stage=max_stage,
            method=method,
            early_stop_marginal=early_stop_marginal,
            patience=patience,
            verbose=verbose)

    def prune_run(self, select=20, max_stage=5, method='omp',
                  n_seeds=3, subsample=0.5):
        """Blind discovery of useful families — no feature-level results exposed.

        Runs auto_run on random subsamples with multiple seeds to identify
        which operation families consistently contribute selected features.
        Returns only family-level info to avoid biasing subsequent runs.

        Parameters
        ----------
        select : int
            Features per selection checkpoint.
        max_stage : int
            Maximum pipeline stage.
        method : str
            Selection method.
        n_seeds : int
            Number of random subsample runs. Families must appear in at
            least one seed to be kept.
        subsample : float
            Fraction of training data per run (0.3–0.8).

        Returns
        -------
        dict with keys:
            'keep' : set of str — families selected in at least one seed
            'exclude' : set of str — families never selected across all seeds
            'family_counts' : dict — {family: n_seeds_where_selected}
        """
        return _workflows.prune_run(
            self, select=select,
            max_stage=max_stage,
            method=method,
            n_seeds=n_seeds,
            subsample=subsample)

    def deepen(self, base_fs, select=25, max_depth=2,
               n_carry=15, decay_stop=0.5,
               marginal_carry_threshold=0.001,
               early_stop_marginal=0.003,
               fm_ks=None, fm_ds=None, pm_ks=None, pm_ps=None, wm_qs=None,
               method='omp', verbose=False):
        """Recursive modulation deepening on selected features.

        Takes a base FeatureSet (typically from auto_run or manual pipeline)
        and builds deeper FM/PM/WM layers on top-carry features with
        thinning grids, similar to RecursiveSoup.

        Parameters
        ----------
        base_fs : FeatureSet
            Starting selection (output of select() or auto_run()).
        select : int
            Features to select at each depth level.
        max_depth : int
            Maximum recursive depth (1 level means one modulation pass).
        n_carry : int
            Number of top features to carry forward.
        decay_stop : float
            Stop if gain(N)/gain(N-1) < this ratio.
        marginal_carry_threshold : float
            Only carry features whose marginal R² exceeds this.
        early_stop_marginal : float
            Stop if gain < this.
        fm_ks, fm_ds, pm_ks, pm_ps, wm_qs : tuple, optional
            Parameter grids.  Defaults from config.
        method : str
            Selection method.
        verbose : bool

        Returns
        -------
        FeatureSet
            Final selected features after deepening.
        """
        return _workflows.deepen(
            self, base_fs=base_fs,
            select=select,
            max_depth=max_depth,
            n_carry=n_carry,
            decay_stop=decay_stop,
            marginal_carry_threshold=marginal_carry_threshold,
            early_stop_marginal=early_stop_marginal,
            fm_ks=fm_ks,
            fm_ds=fm_ds,
            pm_ks=pm_ks,
            pm_ps=pm_ps,
            wm_qs=wm_qs,
            method=method,
            verbose=verbose)

    @atomic_stage
    def fuse(self, *sets):
        """Concatenate multiple FeatureSets without selection.

        Parameters
        ----------
        *sets : FeatureSet
            Feature sets to merge.

        Returns
        -------
        FeatureSet
        """
        if len(sets) == 0:
            raise ValueError("fuse requires at least one FeatureSet")
        # Memory guard for fuse: check total columns before hstack
        if self.max_memory is not None and len(sets) > 1:
            total_cols = sum(s.X.shape[1] for s in sets)
            n_rows = sets[0].X.shape[0]
            if sets[0]._X_val is not None:
                n_rows += sets[0]._X_val.shape[0]
            fuse_bytes = total_cols * n_rows * 8
            if fuse_bytes > self.max_memory:
                raise MemoryBudgetExceeded(
                    'fuse', fuse_bytes, self.max_memory,
                    sets[0].X.shape[0],
                )
            self._memory_high_water = max(self._memory_high_water, fuse_bytes)
        t0 = time.time()
        parent_ids = tuple(self._fs_to_stage[id(s)] for s in sets)
        if len(sets) == 1:
            fs = sets[0]
            self._record('fuse', fs, None, 0, 0,
                         None, None, time.time() - t0, fs.X.shape[1],
                         parent_ids=parent_ids, step_type='fuse')
            return fs
        X = np.hstack([s.X for s in sets])
        names = []
        ft_parts = []
        fam_parts = []
        for s in sets:
            names.extend(s.names)
            ft_parts.append(s.feature_types)
            if s._families is not None:
                fam_parts.extend(s._families)
            else:
                fam_parts.extend(['unknown'] * s.X.shape[1])
        feature_types = np.concatenate(ft_parts)
        # Propagate val: all or nothing
        has_val = all(s._X_val is not None for s in sets)
        X_val = np.hstack([s._X_val for s in sets]) if has_val else None
        fs = FeatureSet(X, names, feature_types, _X_val=X_val,
                        _families=fam_parts)
        input_dims = sum(s.X.shape[1] for s in sets)
        self._record('fuse', fs, None, 0, 0,
                     None, None, time.time() - t0, input_dims,
                     parent_ids=parent_ids, step_type='fuse')
        return fs

    def _select_top(self, cols, names, families, select, method,
                    candidate_types=None, cols_val=None):
        """Build library from generated columns, filter, dedup, select top features.

        Parameters
        ----------
        cols : list of ndarray
            Generated feature columns.
        names : list of str
            Feature names.
        families : list of str
            Family labels.
        select : int or None
            Number of features to select.  When None, skip selection
            entirely and return all surviving features after filter/dedup.
        method : str
            'omp', 'greedy', or 'correlation'.
        candidate_types : list of str, optional
            Per-candidate feature type, parallel to cols/names/families.
            When provided, selected features inherit their type instead of
            defaulting to 'numeric'. Used by raw() to preserve bool/categorical.
        cols_val : list of ndarray, optional
            Validation feature columns, parallel to cols. When provided
            (together with self.y_val), selection scores on val data.

        Returns
        -------
        tuple of (FeatureSet, SelectionHistory or None, int, int, dict or None, list or None)
            ``(feature_set, history, n_generated, n_after_filter, per_class_histories, surviving_names)``
            When select is None, history is None and surviving_names contains
            the names of all features that survived filter/dedup.
        """
        validate_selection(select, method, allow_all=True)
        from ._library import SelectionHistory

        n_train = self.X.shape[0]
        has_val_cols = cols_val is not None
        has_val_score = has_val_cols and self.y_val is not None
        n_generated = len(cols)

        if len(cols) == 0:
            fs = FeatureSet(
                np.empty((n_train, 0)),
                [],
                np.array([], dtype='<U11'),
                _X_val=np.empty((self.X_val.shape[0], 0)) if self.X_val is not None else None,
            )
            return fs, SelectionHistory(), 0, 0, None, None

        X_lib = np.column_stack(cols)

        # Memory guard: post-allocation safety net for select() calls
        # that bypass _guard_candidates
        if self.max_memory is not None:
            alloc_bytes = X_lib.nbytes
            if alloc_bytes > self.max_memory:
                raise MemoryBudgetExceeded(
                    'select', alloc_bytes, self.max_memory,
                    n_train,
                )
            self._memory_high_water = max(self._memory_high_water, alloc_bytes)
        elif X_lib.nbytes > 500 * 1024 * 1024:
            warnings.warn(
                f"Candidate matrix is {X_lib.nbytes // 2**20} MB "
                f"({X_lib.shape[1]:,} candidates × {X_lib.shape[0]:,} samples). "
                f"Consider reducing input dimensions or selection budget.",
                ResourceWarning,
                stacklevel=3,
            )

        X_val_lib = np.column_stack(cols_val) if has_val_cols else None
        lib = PrimitiveLibrary(X_lib, names, families, X_val=X_val_lib)
        lib = lib.filter_constant()
        # Exclude families before correlation deduplication. Otherwise an
        # excluded transform can win a correlated cluster (the default
        # representative is the highest-variance column) and then be removed,
        # taking an allowed raw feature from that cluster with it.
        if self._exclude_families:
            lib = lib.filter_families(self._exclude_families)
        lib = lib.dedup_correlated()
        n_after_filter = lib.X.shape[1]

        # Track all candidate families that survived filter/dedup
        self._candidate_families_seen.update(lib.families)

        if lib.X.shape[1] == 0:
            fs = FeatureSet(
                np.empty((n_train, 0)),
                [],
                np.array([], dtype='<U11'),
                _X_val=np.empty((self.X_val.shape[0], 0)) if self.X_val is not None else None,
            )
            return fs, SelectionHistory(), n_generated, 0, None, None

        # --- select=None: return all surviving features without selection ---
        if select is None:
            surviving = list(lib.names)
            if candidate_types is not None:
                type_lookup = dict(zip(names, candidate_types))
                ft = np.array([type_lookup.get(n, 'numeric') for n in surviving],
                              dtype='<U11')
            else:
                ft = np.full(len(surviving), 'numeric', dtype='<U11')
            X_val_out = lib.X_val if lib.X_val is not None else None
            fs = FeatureSet(lib.X, surviving, ft, _X_val=X_val_out,
                            _families=list(lib.families))
            return fs, None, n_generated, n_after_filter, None, surviving

        # Cap select to available features
        n_select = min(select, lib.X.shape[1])

        history, per_class_histories = select_features(
            lib, self._y_target, n_select, method, task=self.task,
            folds=self.fold_eval_count, seed=self.fold_seed,
            y_val=self._y_val_target if has_val_score else None)

        sel_names = history.selected_names()
        name_to_idx = {n: i for i, n in enumerate(lib.names)}
        sel_idx = [name_to_idx[n] for n in sel_names]
        X_sel = lib.X[:, sel_idx]
        X_val_sel = lib.X_val[:, sel_idx] if lib.X_val is not None else None
        if candidate_types is not None:
            type_lookup = dict(zip(names, candidate_types))
            ft = np.array([type_lookup.get(n, 'numeric') for n in sel_names],
                          dtype='<U11')
        else:
            ft = np.full(len(sel_idx), 'numeric', dtype='<U11')
        sel_families = [lib.families[i] for i in sel_idx]
        fs = FeatureSet(X_sel, sel_names, ft, _X_val=X_val_sel,
                        _families=sel_families)
        return fs, history, n_generated, n_after_filter, per_class_histories, None


SoupPipe = Pipeline  # backward compat alias
