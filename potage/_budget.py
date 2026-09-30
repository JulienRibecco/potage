"""Candidate counts and conservative memory estimates; no feature allocation."""
import numpy as np
from ._records import FeatureSet
from ._feature_utils import _derive_masks
from ._limits import (
    DEDUP_HIERARCHICAL_THRESHOLD as _DEDUP_HIERARCHICAL_THRESHOLD,
    DEDUP_BLOCK_SIZE as _DEDUP_BLOCK_SIZE,
    GRAM_FREE_THRESHOLD as _GRAM_FREE_THRESHOLD,
)

_DEFAULT_MAX_CANDIDATES = 100_000

_SENTINEL_AUTO = 'auto'


def _detect_available_memory():
    """Best-effort detection of available system memory in bytes.

    Strategy:
    1. psutil.virtual_memory().available  (most accurate, cross-platform)
    2. os.sysconf('SC_AVPHYS_PAGES') * page_size  (Linux/macOS, no deps)
    3. Fall back to 2 GB.

    Returns
    -------
    int
        Available memory in bytes.
    """
    # 1. psutil (optional dependency)
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    # 2. os.sysconf (Linux)
    import os
    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (ValueError, OSError, AttributeError):
        pass

    # 3. macOS: vm_stat + sysctl
    import sys
    if sys.platform == 'darwin':
        try:
            import subprocess, re
            out = subprocess.check_output(['vm_stat'],
                                          stderr=subprocess.DEVNULL).decode()
            page_size = int(re.search(r'page size of (\d+)', out).group(1))
            free = int(re.search(r'Pages free:\s+(\d+)', out).group(1))
            inactive = int(re.search(r'Pages inactive:\s+(\d+)', out).group(1))
            spec_m = re.search(r'Pages speculative:\s+(\d+)', out)
            spec = int(spec_m.group(1)) if spec_m else 0
            return (free + inactive + spec) * page_size
        except (OSError, AttributeError, ValueError):
            pass
        # macOS fallback: total memory via sysctl
        try:
            import subprocess
            total = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.memsize'],
                stderr=subprocess.DEVNULL).strip())
            return total // 2  # assume ~50% is available
        except (OSError, ValueError):
            pass

    # 4. Hard fallback
    return 2 * 1024**3


def _resolve_max_memory(value):
    """Resolve a max_memory argument to an integer byte count.

    Parameters
    ----------
    value : int, float, str or None
        - ``'auto'`` — 50% of detected available RAM
        - ``None``   — no limit
        - ``int``    — explicit byte count
        - ``float``  — fraction of detected available RAM (0 < f <= 1)
    """
    if value is None:
        return None
    if isinstance(value, str) and value == 'auto':
        return int(_detect_available_memory() * 0.5)
    if isinstance(value, float) and 0 < value <= 1:
        return int(_detect_available_memory() * value)
    return int(value)


def _estimate_step_peak_bytes(n_candidates, n_total, kfold_k=None):
    """Estimate peak memory for a pipeline step including all allocations.

    Components (all float64 = 8 bytes per element):

    1. Candidate matrix: ``n_candidates × n_total × 8``
       (np.column_stack in _select_top, plus lib.X copy)
    2. Dedup correlation matrix:
       - hierarchical (d < 3000): ``d² × 8``
       - union-find (d ≥ 3000): ``block_size × d × 8``  (block_size=2000)
    3. Selection Gram cost:
       - Gram-free (d ≥ 3000): 0  (on-the-fly computation)
       - Precomputed (d < 3000): ``d² × 8``
    4. K-fold Gram matrices (if kfold_k > 0):
       - d ≥ 3000: auto-switches to correlation, ``1 × d² × 8``
       - d < 3000: ``K × d² × 8``

    Peak = candidate_matrix + max(dedup_cost, selection_cost, kfold_cost)

    Returns
    -------
    dict with keys:
        peak_bytes, candidate_bytes, dedup_bytes, selection_bytes,
        kfold_bytes, dedup_method ('hierarchical' or 'union-find')
    """
    candidate_bytes = n_candidates * n_total * 8

    # Dedup cost
    if n_candidates < _DEDUP_HIERARCHICAL_THRESHOLD:
        dedup_bytes = n_candidates * n_candidates * 8
        dedup_method = 'hierarchical'
    else:
        dedup_bytes = _DEDUP_BLOCK_SIZE * n_candidates * 8
        dedup_method = 'union-find'

    # Selection Gram cost (0 for large d — Gram-free)
    if n_candidates >= _GRAM_FREE_THRESHOLD:
        selection_bytes = 0
    else:
        selection_bytes = n_candidates * n_candidates * 8

    # K-fold cost
    if kfold_k and kfold_k > 1:
        if n_candidates >= _GRAM_FREE_THRESHOLD:
            # Auto-switches to correlation method: 1 accumulated Gram
            kfold_bytes = n_candidates * n_candidates * 8
        else:
            kfold_bytes = kfold_k * n_candidates * n_candidates * 8
    else:
        kfold_bytes = 0

    # Peak is candidate matrix + whichever overhead is larger
    overhead_bytes = max(dedup_bytes, selection_bytes, kfold_bytes)
    peak_bytes = candidate_bytes + overhead_bytes

    return {
        'peak_bytes': peak_bytes,
        'candidate_bytes': candidate_bytes,
        'dedup_bytes': dedup_bytes,
        'selection_bytes': selection_bytes,
        'kfold_bytes': kfold_bytes,
        'dedup_method': dedup_method,
    }


class CandidateBudgetExceeded(RuntimeError):
    """Raised when cumulative candidates across the pipeline exceed max_candidates."""

    def __init__(self, step, step_candidates, max_candidates, used_before,
                 n_samples, breakdown):
        self.step = step
        self.step_candidates = step_candidates
        self.used_before = used_before
        self.max_candidates = max_candidates
        self.n_samples = n_samples
        self.breakdown = breakdown
        top = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ', '.join(f'{k}: {v:,}' for k, v in top)
        new_total = used_before + step_candidates
        super().__init__(
            f"{step}() would push pipeline to ~{new_total:,} total candidates "
            f"({used_before:,} used + {step_candidates:,} new, "
            f"limit: {max_candidates:,}, samples: {n_samples:,}). "
            f"Top families in this step: {top_str}. "
            f"Reduce input dimensions, or set pipe.max_candidates = None to bypass."
        )


class MemoryBudgetExceeded(RuntimeError):
    """Raised when a pipeline step would exceed max_memory bytes."""

    def __init__(self, step, step_bytes, max_memory, n_samples,
                 breakdown_bytes=None):
        self.step = step
        self.step_bytes = step_bytes
        self.max_memory = max_memory
        self.n_samples = n_samples
        self.breakdown_bytes = breakdown_bytes or {}
        step_mb = step_bytes / 2**20
        max_mb = max_memory / 2**20
        parts = []
        if breakdown_bytes:
            for k, v in sorted(breakdown_bytes.items(),
                                key=lambda x: -x[1]):
                if v > 0:
                    parts.append(f'{k}={v // 2**20}MB')
        parts_str = f" [{', '.join(parts)}]" if parts else ''
        super().__init__(
            f"{step}() would allocate ~{step_mb:.0f} MB "
            f"(limit: {max_mb:.0f} MB, samples: {n_samples:,}).{parts_str} "
            f"Reduce input dimensions, or set pipe.max_memory = None to bypass."
        )


class BudgetMixin:
    def predict_candidates(self, step, *inputs, select=None):
        """Estimate candidate count and overfit risk for a pipeline step.

        Returns upper bounds before filter_constant / dedup_correlated pruning.

        Parameters
        ----------
        step : str
            Step name: 'raw', 'am', 'carriers', 'fm', or 'fuse'.
        *inputs : FeatureSet
            Input feature sets (same as the corresponding step method).
            'raw' takes none, 'am' takes one or more, 'carriers' takes one,
            'fm' takes two, 'fuse' takes any number.
        select : int, optional
            Planned selection budget. If given, includes selection pressure
            and effective overfit risk in the output.

        Returns
        -------
        dict with keys:
            candidates : int
                Total candidate features (upper bound).
            breakdown : dict
                Per-family candidate counts.
            n_samples : int
            samples_per_candidate : float
                n_samples / candidates. Higher is safer.
            overfit_risk : str
                'low' (>50), 'moderate' (10-50), 'high' (2-10), 'extreme' (<2).
            select : int or None
                Echo of planned selection budget.
            selection_pressure : float or None
                select / candidates (if select given). Lower = more aggressive.
            effective_ratio : float or None
                n_samples / select (if select given). The ratio that matters
                for the final model.
        """
        n = self.X.shape[0]

        if step == 'raw':
            breakdown = self._predict_raw()
        elif step == 'am':
            if len(inputs) == 0:
                raise ValueError("am requires at least 1 FeatureSet")
            breakdown = self._predict_am(inputs)
        elif step == 'carriers':
            if len(inputs) != 1:
                raise ValueError("carriers requires exactly 1 FeatureSet")
            breakdown = self._predict_carriers(inputs[0])
        elif step == 'fft_carriers':
            if len(inputs) != 1:
                raise ValueError("fft_carriers requires exactly 1 FeatureSet")
            # Conservative estimate: same as carriers (actual count depends
            # on FFT results, but budget check needs an upper bound)
            breakdown = self._predict_carriers(inputs[0])
        elif step == 'fm':
            if len(inputs) != 2:
                raise ValueError("fm requires exactly 2 FeatureSets (carrier, mod)")
            breakdown = self._predict_fm(inputs[0], inputs[1])
        elif step == 'wm':
            if len(inputs) != 2:
                raise ValueError("wm requires exactly 2 FeatureSets (carrier, mod)")
            breakdown = self._predict_wm(inputs[0], inputs[1])
        elif step == 'fm_nonsmooth':
            if len(inputs) != 2:
                raise ValueError("fm_nonsmooth requires exactly 2 FeatureSets (carrier, mod)")
            breakdown = self._predict_fm_nonsmooth(inputs[0], inputs[1])
        elif step == 'fields':
            if len(inputs) != 1:
                raise ValueError("fields requires exactly 1 FeatureSet")
            s = inputs[0]
            is_num, _, _ = _derive_masks(s.feature_types)
            n_num = int(is_num.sum())
            n_pairs = n_num * (n_num - 1) // 2
            n_scales = len(self.config.field_scales)
            # Continuous features: exact count (gauss + inv per scale + nearest)
            # Zone features: theoretical max is n_centers per pair×scale, but
            # most get pruned by zero-variance; use ~30% survival estimate.
            breakdown = {
                'field_continuous': n_pairs * (2 * n_scales + 1),
                'field_zone': n_pairs * n_scales * 5,  # ~30% of centroids survive
                'field_ezone': n_pairs * n_scales * 3,  # fewer survive (maha fallback)
            }
        elif step == 'probe_fields':
            if len(inputs) != 1:
                raise ValueError("probe_fields requires exactly 1 FeatureSet")
            s = inputs[0]
            is_num, _, _ = _derive_masks(s.feature_types)
            n_num = int(is_num.sum())
            n_scales = len(self.config.field_scales)
            # Probe reduces from C(d,2) to sum(C(g_i,2)) over neuron groups.
            # Conservative estimate: assume ~5 groups of ~n_num/5 features each.
            avg_group = max(n_num // 5, 2)
            n_groups = min(5, n_num // 2)
            n_pairs_per_group = avg_group * (avg_group - 1) // 2
            n_pairs = n_groups * n_pairs_per_group
            breakdown = {
                'field_nearest': n_pairs,
                'field_zone': n_pairs * n_scales * 5,
                'field_zcount': n_pairs * n_scales,
            }
        elif step == 'fuse':
            total = sum(s.X.shape[1] for s in inputs)
            breakdown = {'passthrough': total}
        else:
            raise ValueError(f"Unknown step: {step!r}")

        candidates = sum(breakdown.values())
        spc = n / candidates if candidates > 0 else float('inf')

        if spc > 50:
            risk = 'low'
        elif spc > 10:
            risk = 'moderate'
        elif spc > 2:
            risk = 'high'
        else:
            risk = 'extreme'

        remaining = (self.max_candidates - self._candidates_used - candidates
                     if self.max_candidates is not None else None)

        result = {
            'candidates': candidates,
            'breakdown': breakdown,
            'n_samples': n,
            'samples_per_candidate': round(spc, 2),
            'overfit_risk': risk,
            'select': select,
            'selection_pressure': None,
            'effective_ratio': None,
            'candidates_used': self._candidates_used,
            'candidates_after': self._candidates_used + candidates,
            'candidates_remaining': remaining,
        }
        if select is not None and candidates > 0:
            result['selection_pressure'] = round(select / candidates, 4)
            result['effective_ratio'] = round(n / select, 2)

        return result

    def _predict_raw(self):
        """Estimate candidates for raw()."""
        ft = self.feature_types
        d = len(ft)
        d_num = int((ft == 'numeric').sum())
        d_cat = int((ft == 'categorical').sum())

        bd = {}
        bd['raw'] = d
        for fam in ('sq', 'log', 'sqrt', 'cube', 'tanh'):
            bd[fam] = d_num
        bd['reciprocal'] = d_num
        bd['rank'] = d_num
        # Count actual cardinalities for one-hot
        cat_idx = np.where(ft == 'categorical')[0]
        n_onehot = 0
        for i in cat_idx:
            n_onehot += len(np.unique(self.X[:, i][~np.isnan(self.X[:, i])])
                           if np.issubdtype(self.X.dtype, np.floating)
                           else np.unique(self.X[:, i]))
        bd['onehot'] = n_onehot
        return bd

    @staticmethod
    def _type_counts(feature_types):
        ft = np.asarray(feature_types)
        d = len(ft)
        d_num = int((ft == 'numeric').sum())
        d_bool = int((ft == 'bool').sum())
        d_cat = int((ft == 'categorical').sum())
        return d, d_num, d_bool, d_cat

    def _predict_am(self, sets):
        """Estimate candidates for am(*sets)."""
        c2 = lambda n: n * (n - 1) // 2
        max_ratios = self.config.max_ratios_per_feature
        n_ps = len(self.config.power_scales)

        if len(sets) == 1:
            # Self-pairs within one set
            d, d_num, d_bool, d_cat = self._type_counts(sets[0].feature_types)
            d_non_num = d_bool + d_cat
            p_at_least_one_num = c2(d) - c2(d_non_num)
            p_num = c2(d_num)
            if max_ratios is not None:
                n_ratio = d_num * min(d_num - 1, max_ratios) if d_num > 1 else 0
            else:
                n_ratio = d_num * (d_num - 1) if d_num > 1 else 0
        else:
            # Cross-pairs between distinct sets
            dims = [self._type_counts(s.feature_types) for s in sets]
            # Commutative cross-pairs: sum over all (A,B) with A<B of dA*dB
            p_at_least_one_num = 0
            p_num = 0
            n_ratio = 0
            for a in range(len(dims)):
                for b in range(a + 1, len(dims)):
                    da, da_num, da_bool, da_cat = dims[a]
                    db, db_num, db_bool, db_cat = dims[b]
                    da_non = da_bool + da_cat
                    db_non = db_bool + db_cat
                    p_at_least_one_num += da * db - da_non * db_non
                    p_num += da_num * db_num
                    if max_ratios is not None:
                        # Each denom from B, cap numerators from A, and vice versa
                        n_ratio += db_num * min(da_num, max_ratios)
                        n_ratio += da_num * min(db_num, max_ratios)
                    else:
                        n_ratio += 2 * da_num * db_num

        bd = {}
        bd['am_product'] = p_at_least_one_num
        bd['am_diff'] = p_at_least_one_num
        bd['am_ratio'] = n_ratio
        bd['am_sq_diff'] = p_num
        bd['am_min'] = p_num
        bd['am_max'] = p_num
        bd['am_hmean'] = p_num
        bd['am_normdiff'] = p_num
        bd['am_logratio'] = p_num
        bd['am_damped'] = 2 * p_num
        bd['am_relu'] = 2 * p_num
        bd['am_power'] = 2 * n_ps * p_num
        return bd

    def _predict_carriers(self, s):
        """Estimate candidates for carriers(s)."""
        _, d_num, _, _ = self._type_counts(s.feature_types)
        cfg = self.config
        n_freqs = len(cfg.carrier_freqs)

        bd = {}
        bd['carrier_sin'] = n_freqs * d_num
        bd['carrier_cos'] = n_freqs * d_num
        bd['carrier_tri'] = n_freqs * d_num
        bd['carrier_step'] = len(cfg.step_pcts) * d_num
        bd['carrier_pulse'] = len(cfg.pulse_bands) * d_num
        bd['carrier_gauss'] = len(cfg.gauss_center_pcts) * len(cfg.gauss_widths) * d_num
        return bd

    def _predict_fm(self, carrier, mod):
        """Estimate candidates for fm(carrier, mod)."""
        _, d_c_num, _, _ = self._type_counts(carrier.feature_types)
        d_m, _, _, _ = self._type_counts(mod.feature_types)
        cfg = self.config
        n_freqs = len(cfg.carrier_freqs)
        n_fm_d = len(cfg.fm_depths)
        n_pm_d = len(cfg.pm_depths)

        # Cross-set: numeric carriers from carrier set × all mods from mod set
        pairs = d_c_num * d_m

        bd = {}
        bd['fm_sin'] = pairs * n_freqs * n_fm_d
        bd['pm_sin'] = pairs * n_freqs * n_pm_d
        return bd

    def _predict_wm(self, carrier, mod):
        """Estimate candidates for wm(carrier, mod)."""
        _, d_c_num, _, _ = self._type_counts(carrier.feature_types)
        d_m, _, _, _ = self._type_counts(mod.feature_types)
        cfg = self.config
        pairs = d_c_num * d_m
        n_center = len(cfg.wm_center_pcts)
        n_width = n_center * len(cfg.wm_width_mods)
        bd = {}
        bd['wm_center'] = pairs * n_center
        bd['wm_width'] = pairs * n_width
        return bd

    def _predict_fm_nonsmooth(self, carrier, mod):
        """Estimate candidates for fm_nonsmooth(carrier, mod)."""
        _, d_c_num, _, _ = self._type_counts(carrier.feature_types)
        d_m, _, _, _ = self._type_counts(mod.feature_types)
        cfg = self.config
        pairs = d_c_num * d_m
        n_freqs = len(cfg.carrier_freqs)
        n_fm_d = len(cfg.fm_depths)
        n_gauss = len(cfg.gauss_center_pcts) * len(cfg.gauss_widths)
        bd = {}
        bd['fm_tri'] = pairs * n_freqs * n_fm_d
        bd['fm_pulse'] = pairs * n_gauss
        return bd

    def estimate_memory(self, step, *inputs):
        """Estimate peak memory for a pipeline step.

        Accounts for all allocations: candidate matrix, dedup
        correlation/block matrix, and K-fold Gram matrices.

        Parameters
        ----------
        step : str
            Step name (same as predict_candidates).
        *inputs : FeatureSet
            Input feature sets.

        Returns
        -------
        dict with keys:
            peak_bytes : int
                Estimated peak allocation in bytes.
            candidate_bytes : int
                Candidate matrix alone (n_candidates × n_total × 8).
            dedup_bytes : int
                Dedup correlation matrix cost.
            selection_bytes : int
                Selection Gram matrix cost (0 for gram-free).
            kfold_bytes : int
                K-fold Gram matrix cost (0 if no kfold).
            dedup_method : str
                'hierarchical' or 'union-find'.
            candidates : int
            breakdown_bytes : dict
                Per-family byte estimates (candidate matrix only).
            n_samples : int
        """
        est = self.predict_candidates(step, *inputs)
        n_total = est['n_samples']
        if self.X_val is not None:
            n_total += self.X_val.shape[0]
        kfold_k = (self.fold_eval_count
                    if self.fold_eval_count and self.fold_eval_count > 1
                    and self.y_val is None
                    else None)
        mem = _estimate_step_peak_bytes(est['candidates'], n_total, kfold_k)
        breakdown_bytes = {k: v * n_total * 8
                           for k, v in est['breakdown'].items()}
        return {
            'peak_bytes': mem['peak_bytes'],
            'candidate_bytes': mem['candidate_bytes'],
            'dedup_bytes': mem['dedup_bytes'],
            'selection_bytes': mem['selection_bytes'],
            'kfold_bytes': mem['kfold_bytes'],
            'dedup_method': mem['dedup_method'],
            'candidates': est['candidates'],
            'breakdown_bytes': breakdown_bytes,
            'n_samples': est['n_samples'],
        }

    def print_memory(self, step, *inputs):
        """Print human-readable memory estimate for a pipeline step.

        Convenience wrapper around :meth:`estimate_memory` that formats
        bytes as MB/GB and shows budget status.

        Example::

            pipe.print_memory('fm', fs_raw, fs_raw)
            # fm: 4968 MB peak (budget: 1024 MB) — EXCEEDS BUDGET
            #   candidates: 123 MB, kfold: 4844 MB, dedup: 384 MB
        """
        est = self.estimate_memory(step, *inputs)

        def _fmt(b):
            if b >= 1024 ** 3:
                return f"{b / 1024**3:.1f} GB"
            return f"{b / 1024**2:.0f} MB"

        peak = _fmt(est['peak_bytes'])
        budget = _fmt(self.max_memory) if self.max_memory else 'unlimited'
        fits = self.max_memory is None or est['peak_bytes'] <= self.max_memory
        status = "OK" if fits else "EXCEEDS BUDGET"

        print(f"  {step}: {peak} peak (budget: {budget}) — {status}")
        print(f"    {est['candidates']} candidates, "
              f"kfold: {_fmt(est['kfold_bytes'])}, "
              f"dedup: {_fmt(est['dedup_bytes'])}, "
              f"selection: {_fmt(est['selection_bytes'])}")

    def predict_pipeline_memory(self, max_stage=5, select=20):
        """Simulate auto_run's DAG and predict per-stage memory.

        Does not build any features — uses :meth:`predict_candidates`
        with mock dimension counts to estimate peak allocation at
        each stage and the fuse pool size.

        Parameters
        ----------
        max_stage : int
            Maximum stage (0–5), same as auto_run.
        select : int
            Selection budget per checkpoint, same as auto_run.

        Returns
        -------
        dict with keys:
            stages : list of dict
                Per-stage info: name, candidates, peak_bytes,
                pool_bytes_after, exceeds_budget.
            total_peak_bytes : int
                Maximum single-step allocation across all stages.
            fits_in_budget : bool
            recommended_max_stage : int
                Highest stage that fits within max_memory.
        """
        n_train = self.X.shape[0]
        n_val = self.X_val.shape[0] if self.X_val is not None else 0
        n_total = n_train + n_val
        max_mem = self.max_memory
        kfold_k = (self.fold_eval_count
                    if self.fold_eval_count and self.fold_eval_count > 1
                    and self.y_val is None
                    else None)

        stages_info = []
        overall_peak = 0
        recommended = 0
        pool_cols = 0  # running count of pool columns

        def _stage_entry(name, cands, pool_cols_after):
            """Build a stage info dict with full memory estimation."""
            mem = _estimate_step_peak_bytes(cands, n_total, kfold_k)
            fuse_bytes = pool_cols_after * n_total * 8
            exceeds = (max_mem is not None
                       and (mem['peak_bytes'] > max_mem
                            or fuse_bytes > max_mem))
            return {
                'name': name,
                'candidates': cands,
                'peak_bytes': mem['peak_bytes'],
                'candidate_bytes': mem['candidate_bytes'],
                'dedup_bytes': mem['dedup_bytes'],
                'selection_bytes': mem['selection_bytes'],
                'kfold_bytes': mem['kfold_bytes'],
                'dedup_method': mem['dedup_method'],
                'pool_bytes_after': fuse_bytes,
                'exceeds_budget': exceeds,
            }, max(mem['peak_bytes'], fuse_bytes)

        # Helper to build a mock FeatureSet with given dimensions
        def _mock_fs(d, d_num=None):
            if d_num is None:
                d_num = d
            ft = np.array(['numeric'] * d_num + ['bool'] * (d - d_num),
                           dtype='<U11')
            return FeatureSet(
                np.empty((n_train, d)),
                [f'_mock_{i}' for i in range(d)],
                ft,
                _X_val=np.empty((n_val, d)) if n_val > 0 else None,
            )

        # Stage 0: raw
        raw_bd = self._predict_raw()
        raw_cands = sum(raw_bd.values())
        # After raw select=None, approximate surviving features as raw_cands
        # (filter/dedup typically removes ~20-40%, but we use upper bound)
        raw_out_cols = raw_cands
        pool_cols = raw_out_cols
        entry, stage_peak = _stage_entry('raw', raw_cands, pool_cols)
        stages_info.append(entry)
        overall_peak = max(overall_peak, stage_peak)
        if not entry['exceeds_budget']:
            recommended = 0

        if max_stage < 1:
            return self._pipeline_memory_result(
                stages_info, overall_peak, max_mem, recommended)

        # Stage 1: AM on raw output
        mock_s0 = _mock_fs(raw_out_cols)
        am_bd = self._predict_am((mock_s0,))
        am_cands = sum(am_bd.values())
        pool_cols += am_cands
        entry, stage_peak = _stage_entry('am', am_cands, pool_cols)
        stages_info.append(entry)
        overall_peak = max(overall_peak, stage_peak)
        if not entry['exceeds_budget']:
            recommended = 1
        pool_cols = select

        if max_stage < 2:
            return self._pipeline_memory_result(
                stages_info, overall_peak, max_mem, recommended)

        # Stage 2: carriers on raw output
        carriers_bd = self._predict_carriers(mock_s0)
        carriers_cands = sum(carriers_bd.values())
        pool_cols += carriers_cands
        entry, stage_peak = _stage_entry('carriers', carriers_cands, pool_cols)
        stages_info.append(entry)
        overall_peak = max(overall_peak, stage_peak)
        if not entry['exceeds_budget']:
            recommended = 2
        pool_cols = select

        # Stages 3-5: modulation stages use carrier (raw_out) × modulator (select)
        mod_stages = [
            (3, 'fm', self._predict_fm),
            (4, 'fm_nonsmooth', self._predict_fm_nonsmooth),
            (5, 'wm', self._predict_wm),
        ]
        mock_mod = _mock_fs(select)
        for stage_num, name, predict_fn in mod_stages:
            if max_stage < stage_num:
                break
            bd = predict_fn(mock_s0, mock_mod)
            cands = sum(bd.values())
            pool_cols += cands
            entry, stage_peak = _stage_entry(name, cands, pool_cols)
            stages_info.append(entry)
            overall_peak = max(overall_peak, stage_peak)
            if not entry['exceeds_budget']:
                recommended = stage_num
            pool_cols = select

        return self._pipeline_memory_result(
            stages_info, overall_peak, max_mem, recommended)

    @staticmethod
    def _pipeline_memory_result(stages_info, overall_peak, max_mem,
                                recommended):
        fits = max_mem is None or overall_peak <= max_mem
        return {
            'stages': stages_info,
            'total_peak_bytes': overall_peak,
            'fits_in_budget': fits,
            'recommended_max_stage': recommended,
        }


def estimate_candidates(d, d_num=None, d_bool=0, d_cat=0, max_stage=2,
                        carrier_freqs=None, max_ratios=5,
                        power_scales=None, pointwise_only=True):
    """Estimate total soup candidates from pipeline config (no data needed).

    Returns dict with per-stage counts and total.  These are upper bounds
    before filter_constant / dedup_correlated pruning.

    Parameters
    ----------
    d : int
        Total number of input features.
    d_num : int, optional
        Number of numeric features.  Defaults to ``d - d_bool - d_cat``
        (or *d* if no type counts given).
    d_bool : int
        Number of boolean features.
    d_cat : int
        Number of categorical features.
    max_stage : int
        Highest stage to include (0, 1, or 2).
    carrier_freqs : list of float, optional
        Frequency grid for stage 2 carriers.  Default ``[0.5, 1.0, 2.0, 4.0]``
        (ResidualSoup default).
    max_ratios : int or None
        Cap on ratio features per denominator.  None = unlimited.
    power_scales : list of float, optional
        Exponent multipliers for power features.  Default ``[0.5, 1.0, 2.0, 3.0]``.
    pointwise_only : bool
        If True (default), exclude distribution-dependent families
        (rank from stage 0; step/pulse/gauss from stage 2).
    """
    if d_num is None:
        d_num = d - d_bool - d_cat
    if d_num + d_bool + d_cat > d:
        raise ValueError(
            f"d_num ({d_num}) + d_bool ({d_bool}) + d_cat ({d_cat}) "
            f"cannot exceed d ({d})")
    if carrier_freqs is None:
        carrier_freqs = [0.5, 1.0, 2.0, 3.0]
    if power_scales is None:
        power_scales = [0.5, 1.0, 2.0, 3.0]

    n_freqs = len(carrier_freqs)
    n_ps = len(power_scales)

    def _c2(n):
        """Combinations C(n, 2)."""
        return n * (n - 1) // 2

    # ------------------------------------------------------------------
    # Stage 0: unary transforms (numeric only) + raw passthrough (all)
    # ------------------------------------------------------------------
    s0 = {}
    s0['raw'] = d
    for family in ('sq', 'log', 'sqrt', 'cube', 'tanh'):
        s0[family] = d_num
    s0['reciprocal'] = d_num
    if not pointwise_only:
        s0['rank'] = d_num
    # one-hot: categorical only (rough estimate — actual depends on cardinality)
    s0['onehot'] = d_cat * 5  # rough upper bound
    stage_0 = sum(s0.values())

    # ------------------------------------------------------------------
    # Stage 1: pairwise AM ops (at-least-one-numeric rule for products/diff)
    # ------------------------------------------------------------------
    s1 = {}
    p_num = _c2(d_num)
    d_non_num = d_bool + d_cat
    # Pairs with at least one numeric: C(d,2) - C(d_non_num, 2)
    p_at_least_one_num = _c2(d) - _c2(d_non_num)

    s1['products'] = p_at_least_one_num
    s1['abs_diff'] = p_at_least_one_num
    if max_ratios is not None:
        s1['ratios'] = d_num * min(d_num - 1, max_ratios) if d_num > 0 else 0
    else:
        s1['ratios'] = d_num * (d_num - 1) if d_num > 0 else 0
    s1['sq_diff'] = p_num
    s1['min'] = p_num
    s1['max'] = p_num
    s1['hmean'] = p_num
    s1['normdiff'] = p_num
    s1['logratio'] = p_num
    s1['damped'] = 2 * p_num
    s1['relu'] = 2 * p_num
    s1['power'] = 2 * n_ps * p_num
    stage_1 = sum(s1.values())

    # ------------------------------------------------------------------
    # Stage 2: static carriers on numeric features
    # ------------------------------------------------------------------
    s2 = {}
    s2['sin'] = n_freqs * d_num
    s2['cos'] = n_freqs * d_num
    s2['tri'] = n_freqs * d_num
    if not pointwise_only:
        s2['step'] = 3 * d_num       # default 3 step_pcts
        s2['pulse'] = 3 * d_num      # default 3 pulse_bands
        s2['gauss'] = 3 * 2 * d_num  # default 3 centers × 2 widths
    stage_2 = sum(s2.values())

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    stages = {'stage_0': stage_0}
    total = stage_0
    breakdown = {'stage_0': s0}

    if max_stage >= 1:
        stages['stage_1'] = stage_1
        total += stage_1
        breakdown['stage_1'] = s1
    if max_stage >= 2:
        stages['stage_2'] = stage_2
        total += stage_2
        breakdown['stage_2'] = s2

    return {**stages, 'total': total, 'breakdown': breakdown}
