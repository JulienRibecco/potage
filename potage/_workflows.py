"""Discovery strategies composed from pipeline operations.

These ordinary functions receive a Pipeline. Public convenience methods on
Pipeline preserve the existing API while orchestration lives here.
"""

from ._modulation import (
    _apply_modulation,
    _fm_sin_kernel,
    _pm_sin_kernel,
    _wm_center_kernel,
    _wm_width_kernel,
)
from ._carriers import _space_label
from ._records import FeatureSet
from ._budget import CandidateBudgetExceeded, MemoryBudgetExceeded
import numpy as np
from collections import Counter


def _thin_grid(grid, level, base_level=2):
    """Thin a parameter grid for deeper modulation levels.

    At base_level, returns full grid. Each level deeper removes ~29% of
    elements (sqrt(2) thinning), keeping more resolution at depth than the
    previous 2x step. Grids with <=2 elements are never thinned.
    """
    grid = list(grid)
    if len(grid) <= 2 or level <= base_level:
        return grid
    # sqrt(2) thinning: keep fraction 1/sqrt(2) per level beyond base
    depth = level - base_level
    keep_frac = (1.0 / 1.4142) ** depth
    keep_n = max(2, int(round(len(grid) * keep_frac)))
    if keep_n >= len(grid):
        return grid
    # Evenly spaced indices
    indices = np.linspace(0, len(grid) - 1, keep_n).round().astype(int)
    return [grid[i] for i in np.unique(indices)]


def discover_carrier_space(pipe, fs_raw, select=15, method='omp',
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
    from ._carriers import _space_label

    if spaces is None:
        spaces = ['linear', 'arcsinh', 'log', 'sqrt', 'cbrt', 'tanh']

    sets = []
    if include_raw:
        sets.append(fs_raw)

    skipped = {}
    for sp in spaces:
        try:
            fs_sp = pipe.carriers(fs_raw, select=select, method=method,
                                  feature_space=sp)
            sets.append(fs_sp)
        except (CandidateBudgetExceeded, MemoryBudgetExceeded) as exc:
            skipped[sp] = str(exc)

    fused = pipe.fuse(*sets)
    fs_final = pipe.select(fused, select=select, method=method)

    # Count winners per space by inspecting feature name prefixes
    counts = Counter()
    for name in fs_final.names:
        matched = 'raw'
        for sp in spaces:
            lbl = _space_label(sp)
            if lbl and name.startswith(f'{lbl}('):
                matched = sp
                break
            # arcsinh/tanh/logit carrier names contain the label inside
            if lbl and lbl in name:
                matched = sp
                break
        counts[matched] += 1

    meta = {'winner_counts': dict(counts), 'spaces': spaces, 'skipped': skipped}
    return fs_final, meta


def auto_run(pipe, select=20, max_stage=5, method='omp',
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
    from ._diagnose import diagnose_curve
    if method == 'correlation' or pipe.task != 'regression':
        raise ValueError("auto_run uses R² stopping; choose regression with omp or greedy")

    # Upfront memory check: reduce max_stage if pipeline won't fit
    if pipe.max_memory is not None:
        profile = pipe.predict_pipeline_memory(max_stage, select)
        if not profile['fits_in_budget']:
            old_max = max_stage
            max_stage = profile['recommended_max_stage']
            if verbose:
                print(f'auto_run: memory budget would be exceeded at '
                      f'stage {old_max}, reducing to max_stage={max_stage}')

    consecutive_below = 0
    prev_r2 = 0.0

    # Stage 0: raw features (no selection)
    if verbose:
        print('auto_run: stage 0 (raw)')
    s0 = pipe.raw(select=None)
    pool = s0
    # Reduced raw set for budget-constrained stages
    s0_reduced = None

    def _get_s0_reduced():
        nonlocal s0_reduced
        if s0_reduced is None:
            if verbose:
                print('  budget exceeded, pre-selecting raw features')
            s0_reduced = pipe.select(s0, select=select * 3, method=method)
        return s0_reduced

    # Stage 1: AM
    if max_stage >= 1:
        if verbose:
            print('auto_run: stage 1 (AM)')
        try:
            s1 = pipe.am(s0, select=None)
        except (CandidateBudgetExceeded, MemoryBudgetExceeded):
            s1 = pipe.am(_get_s0_reduced(), select=None)
        pool = pipe.fuse(pool, s1)

    # Select from pool so far
    result = pipe.select(pool, select=select, method=method)

    # Check early stopping
    stage_obj = pipe._stages[-1]
    r2 = stage_obj.final_r2
    if r2 is not None:
        gain = r2 - prev_r2
        if verbose:
            print(f'  -> R²={r2:.4f} (+{gain:.4f})')
        if gain < early_stop_marginal:
            consecutive_below += 1
        else:
            consecutive_below = 0
        prev_r2 = r2
        if consecutive_below >= patience:
            if verbose:
                print(f'  early stop: {consecutive_below} stages below threshold')
            return result

    # Stage 2: carriers
    if max_stage >= 2:
        if verbose:
            print('auto_run: stage 2 (carriers)')
        try:
            s2 = pipe.carriers(s0, select=None)
        except (CandidateBudgetExceeded, MemoryBudgetExceeded):
            s2 = pipe.carriers(_get_s0_reduced(), select=None)
        pool = pipe.fuse(pool, s2)
        result = pipe.select(pool, select=select, method=method)
        stage_obj = pipe._stages[-1]
        r2 = stage_obj.final_r2
        if r2 is not None:
            gain = r2 - prev_r2
            if verbose:
                print(f'  -> R²={r2:.4f} (+{gain:.4f})')
            if gain < early_stop_marginal:
                consecutive_below += 1
            else:
                consecutive_below = 0
            prev_r2 = r2
            if consecutive_below >= patience:
                if verbose:
                    print(f'  early stop: {consecutive_below} stages below threshold')
                return result

    # Stages 3-5 use carrier × modulator patterns.
    # When budget is exceeded, use reduced raw set as carrier.
    def _try_mod_stage(stage_name, method_fn, carrier, mod):
        """Try modulation stage, fallback to reduced carrier on budget overflow."""
        try:
            return method_fn(carrier, mod, select=None)
        except (CandidateBudgetExceeded, MemoryBudgetExceeded):
            reduced = _get_s0_reduced()
            if verbose:
                print(f'  budget exceeded for {stage_name}, using reduced carrier')
            return method_fn(reduced, mod, select=None)

    # Stage 3: FM/PM sin
    if max_stage >= 3:
        if verbose:
            print('auto_run: stage 3 (FM sin)')
        s3 = _try_mod_stage('FM', pipe.fm, s0, result)
        pool = pipe.fuse(pool, s3)
        result = pipe.select(pool, select=select, method=method)
        stage_obj = pipe._stages[-1]
        r2 = stage_obj.final_r2
        if r2 is not None:
            gain = r2 - prev_r2
            if verbose:
                print(f'  -> R²={r2:.4f} (+{gain:.4f})')
            if gain < early_stop_marginal:
                consecutive_below += 1
            else:
                consecutive_below = 0
            prev_r2 = r2
            if consecutive_below >= patience:
                if verbose:
                    print(f'  early stop: {consecutive_below} stages below threshold')
                return result

    # Stage 4: FM/PM nonsmooth
    if max_stage >= 4:
        if verbose:
            print('auto_run: stage 4 (FM nonsmooth)')
        s4 = _try_mod_stage('FM_nonsmooth', pipe.fm_nonsmooth, s0, result)
        pool = pipe.fuse(pool, s4)
        result = pipe.select(pool, select=select, method=method)
        stage_obj = pipe._stages[-1]
        r2 = stage_obj.final_r2
        if r2 is not None:
            gain = r2 - prev_r2
            if verbose:
                print(f'  -> R²={r2:.4f} (+{gain:.4f})')
            if gain < early_stop_marginal:
                consecutive_below += 1
            else:
                consecutive_below = 0
            prev_r2 = r2
            if consecutive_below >= patience:
                if verbose:
                    print(f'  early stop: {consecutive_below} stages below threshold')
                return result

    # Stage 5: WM
    if max_stage >= 5:
        if verbose:
            print('auto_run: stage 5 (WM)')
        s5 = _try_mod_stage('WM', pipe.wm, s0, result)
        pool = pipe.fuse(pool, s5)
        result = pipe.select(pool, select=select, method=method)
        if verbose:
            stage_obj = pipe._stages[-1]
            r2 = stage_obj.final_r2
            if r2 is not None:
                print(f'  -> R²={r2:.4f}')

    # Report unused families
    if verbose:
        unused = pipe.unused_families()
        if unused:
            print(f'  unused families: {", ".join(sorted(unused))}'
                  f'  (consider excluding to free budget)')

    return result


def prune_run(pipe, select=20, max_stage=5, method='omp',
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
    from ._pipe import Pipeline
    n = pipe.X.shape[0]
    n_sub = max(50, int(n * subsample))

    family_counts = {}
    all_seen = set()

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + 7)
        idx = rng.choice(n, size=n_sub, replace=False)

        sub_pipe = Pipeline(
            pipe.X[idx], pipe._y_target[idx], pipe.names,
            feature_types=pipe.feature_types,
            config=pipe.config,
            max_candidates=pipe.max_candidates,
            max_memory=pipe.max_memory,
            task=pipe.task,
        )
        sub_pipe.auto_run(select=select, max_stage=max_stage, method=method)

        all_seen.update(sub_pipe._candidate_families_seen)
        used = sub_pipe._candidate_families_seen - sub_pipe.unused_families()
        for fam in used:
            family_counts[fam] = family_counts.get(fam, 0) + 1

    keep = set(family_counts.keys())
    exclude = all_seen - keep

    return {
        'keep': keep,
        'exclude': exclude,
        'family_counts': family_counts,
    }


def deepen(pipe, base_fs, select=25, max_depth=2,
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
    if method == 'correlation' or pipe.task != 'regression':
        raise ValueError("deepen uses R² stopping; choose regression with omp or greedy")
    cfg = pipe.config
    if fm_ks is None:
        fm_ks = tuple(cfg.carrier_freqs)
    if fm_ds is None:
        fm_ds = tuple(cfg.fm_depths)
    if pm_ks is None:
        pm_ks = tuple(cfg.carrier_freqs)
    if pm_ps is None:
        pm_ps = tuple(cfg.pm_depths)
    if wm_qs is None:
        wm_qs = tuple(cfg.wm_center_pcts)

    # Get the base selection's history for marginal R² info
    base_stage_idx = pipe._fs_to_stage.get(id(base_fs))
    base_stage = pipe._stages[base_stage_idx] if base_stage_idx is not None else None

    # Accumulate pool starting with the parent of base_fs
    # (all features available before the select that produced base_fs)
    pool_fs = base_fs
    result = base_fs
    prev_r2 = base_stage.final_r2 if base_stage and base_stage.final_r2 else 0.0
    prev_gain = None

    for depth in range(2, max_depth + 2):
        if verbose:
            print(f'deepen: depth {depth}')

        # Extract carry features from current selection
        if base_stage and base_stage.history is not None:
            sel_names = base_stage.history.selected_names()
            marginals = base_stage.history.marginal_r2()
            carry_names = []
            for i, (name, marg) in enumerate(zip(sel_names, marginals)):
                if i < n_carry and marg >= marginal_carry_threshold:
                    carry_names.append(name)
        else:
            carry_names = result.names[:n_carry]

        if len(carry_names) == 0:
            if verbose:
                print('  no features to carry, stopping')
            break

        # Build carry FeatureSet
        name_to_idx = {n: i for i, n in enumerate(result.names)}
        carry_idx = [name_to_idx[n] for n in carry_names if n in name_to_idx]
        carry_found = [n for n in carry_names if n in name_to_idx]
        if len(carry_found) == 0:
            if verbose:
                print('  no carry features found, stopping')
            break

        X_carry = result.X[:, carry_idx]
        X_carry_val = result._X_val[:, carry_idx] if result._X_val is not None else None
        carry_types = np.full(len(carry_found), 'numeric', dtype='<U11')
        carry_fs = FeatureSet(X_carry, carry_found, carry_types,
                              _X_val=X_carry_val)
        # Register carry_fs so it gets a stage index
        pipe._record(f'carry_L{depth}', carry_fs, None, 0, 0,
                     None, None, 0.0, len(carry_found),
                     parent_ids=(pipe._fs_to_stage[id(result)],),
                     step_type='select', surviving_names=carry_found)

        # Thin parameter grids
        level_fm_ks = _thin_grid(fm_ks, depth)
        level_fm_ds = _thin_grid(fm_ds, depth)
        level_pm_ks = _thin_grid(pm_ks, depth)
        level_pm_ps = _thin_grid(pm_ps, depth)
        level_wm_qs = _thin_grid(wm_qs, depth)

        if verbose:
            print(f'  carry: {len(carry_found)} features')
            print(f'  grids: FM k={level_fm_ks} d={level_fm_ds}')

        # Build self-modulation using _apply_modulation (i!=j exclusion)
        is_numeric = np.ones(len(carry_found), dtype=bool)
        top_idx = np.arange(len(carry_found))

        fm_grid = [{'k': k, 'd': d}
                   for k in level_fm_ks for d in level_fm_ds]
        pm_grid = [{'k': k, 'p': p}
                   for k in level_pm_ks for p in level_pm_ps]
        wm_center_grid = [{'q': q} for q in level_wm_qs]
        wm_width_grid = [{'q': q, 'wmod': wmod}
                         for q in level_wm_qs
                         for wmod in cfg.wm_width_mods]

        kernels = [
            (_fm_sin_kernel, fm_grid, f'L{depth}_fm'),
            (_pm_sin_kernel, pm_grid, f'L{depth}_pm'),
            (_wm_center_kernel, wm_center_grid, f'L{depth}_wm_center'),
            (_wm_width_kernel, wm_width_grid, f'L{depth}_wm_width'),
        ]

        new_cols, new_names, new_fams = _apply_modulation(
            X_carry, carry_found, is_numeric, cfg, top_idx, kernels)

        # Build val modulation
        new_val_cols = None
        if X_carry_val is not None:
            new_val_cols, _, _ = _apply_modulation(
                X_carry_val, carry_found, is_numeric, cfg, top_idx,
                kernels, reference_X=X_carry)

        if verbose:
            print(f'  new features: {len(new_cols)}')

        if len(new_cols) == 0:
            if verbose:
                print('  no modulation features generated, stopping')
            break

        # Create FeatureSet from new features
        X_new = np.column_stack(new_cols)
        X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_new = None
        if new_val_cols is not None:
            X_val_new = np.column_stack(new_val_cols)
            X_val_new = np.nan_to_num(X_val_new, nan=0.0, posinf=0.0, neginf=0.0)
        new_ft = np.full(len(new_names), 'numeric', dtype='<U11')
        new_fs = FeatureSet(X_new, new_names, new_ft, _X_val=X_val_new)
        # Register new_fs with special step_type for transform replay
        pipe._record(f'deepen_L{depth}_gen', new_fs, None,
                     len(new_cols), len(new_cols),
                     None, None, 0.0, len(carry_found),
                     parent_ids=(pipe._fs_to_stage[id(carry_fs)],),
                     step_type='deepen_gen',
                     surviving_names=list(new_names),
                     metadata={'depth': depth,
                               'fm_ks': level_fm_ks,
                               'fm_ds': level_fm_ds,
                               'pm_ks': level_pm_ks,
                               'pm_ps': level_pm_ps,
                               'wm_qs': level_wm_qs})

        # Fuse with accumulated pool
        pool_fs = pipe.fuse(pool_fs, new_fs)

        # Select from combined pool
        result = pipe.select(pool_fs, select=select, method=method)

        stage_obj = pipe._stages[-1]
        curr_r2 = stage_obj.final_r2 or 0.0
        gain = curr_r2 - prev_r2

        if verbose:
            print(f'  -> R²={curr_r2:.4f} (+{gain:.4f})')

        # Decay stopping
        if prev_gain is not None and prev_gain > 0:
            decay = gain / prev_gain
            if verbose:
                print(f'  decay={decay:.2f}')
            if decay < decay_stop:
                if verbose:
                    print(f'  stopping: decay {decay:.2f} < {decay_stop}')
                break

        if gain < early_stop_marginal:
            if verbose:
                print(f'  stopping: marginal gain {gain:.4f} < {early_stop_marginal}')
            break

        prev_r2 = curr_r2
        prev_gain = gain
        base_stage = stage_obj

    return result
