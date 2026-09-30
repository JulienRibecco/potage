"""Localized field features, zone adjustments, and optional neural grouping."""

from ._provenance import derived_name, input_names, joined_name
import numpy as np


_DEFAULT_PCT_BY_DIM = {
    2: [20, 40, 60, 80],     # 4 levels → 16 centers
    3: [25, 50, 75],          # 3 levels → 27 centers
    4: [33, 67],              # 2 levels → 16 centers
}


def _build_zones_for_group(pts, centers, col_ids, group_name,
                           scales, dedup, reference_pts=None):
    """Generate circle/sphere zone features for one column group."""
    cols, pnames, families = [], [], []
    fset = frozenset(col_ids)
    n_centers = len(centers)

    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    dists = np.sqrt(d2)
    reference_dists = (dists if reference_pts is None else np.sqrt(
        ((reference_pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)))

    fp_near = ('field_near', fset)
    if dedup is None or dedup.is_new(fp_near, 'field_nearest'):
        cols.append(dists.min(axis=1))
        pnames.append(derived_name(f'field_near({group_name})', group_name))
        families.append('field_nearest')

    for sigma in scales:
        in_zone = (dists <= sigma).astype(np.float64)
        reference_zone = (reference_dists <= sigma).astype(np.float64)
        for k in range(n_centers):
            if reference_zone[:, k].std() < 1e-10:
                continue
            fp_z = ('field_zone', fset, k, round(sigma, 6))
            if dedup is None or dedup.is_new(fp_z, 'field_zone'):
                cols.append(in_zone[:, k])
                pnames.append(derived_name(f'zone({group_name},c{k},r={sigma})', group_name))
                families.append('field_zone')

        fp_zc = ('field_zcount', fset, round(sigma, 6))
        if dedup is None or dedup.is_new(fp_zc, 'field_zcount'):
            zcount = in_zone.sum(axis=1)
            if reference_zone.sum(axis=1).std() > 1e-10:
                cols.append(zcount)
                pnames.append(derived_name(f'zcount({group_name},r={sigma})', group_name))
                families.append('field_zcount')

    return cols, pnames, families


def build_stage_fields(X, names, numeric_mask, config=None, dedup=None,
                       reference_X=None):
    """Build zone features from column-group percentile grids.

    For groups of 2, 3, or 4 numeric columns, normalises to [0,1],
    builds a percentile grid, and tests zone membership via Euclidean
    distance.  Continuous fields (Gaussian/inverse) and Mahalanobis
    ellipses removed — redundant with AM × carriers.

    Returns (cols, pnames, families).
    """
    names = input_names(names)
    from itertools import combinations, product as iprod

    cols, pnames, families = [], [], []

    scales = [0.1, 0.3, 1.0]
    max_dims = 2
    custom_pct = None
    if config is not None:
        scales = getattr(config, 'field_scales', scales)
        max_dims = getattr(config, 'field_max_dims', max_dims)
        custom_pct = getattr(config, 'field_pct_levels', None)

    num_idx = np.where(numeric_mask)[0]
    if len(num_idx) < 2:
        return cols, pnames, families

    ref = reference_X if reference_X is not None else X

    ref_mins = ref[:, num_idx].min(axis=0)
    ref_maxs = ref[:, num_idx].max(axis=0)
    ref_ranges = ref_maxs - ref_mins
    ref_ranges[ref_ranges < 1e-10] = 1.0

    X_norm = (X[:, num_idx] - ref_mins) / ref_ranges
    ref_norm = (ref[:, num_idx] - ref_mins) / ref_ranges

    n_num = len(num_idx)

    for ndim in range(2, min(max_dims, n_num) + 1):
        if ndim == 2 and custom_pct is not None:
            pct_levels = custom_pct
        else:
            pct_levels = _DEFAULT_PCT_BY_DIM.get(ndim, [33, 67])

        col_pcts = {}
        for li in range(n_num):
            col_pcts[li] = np.percentile(ref_norm[:, li], pct_levels)

        for group in combinations(range(n_num), ndim):
            col_ids = tuple(int(num_idx[g]) for g in group)
            group_name = joined_name('+', (names[c] for c in col_ids))

            grid_axes = [col_pcts[g] for g in group]
            centers = np.array(list(iprod(*grid_axes)))

            pts = X_norm[:, list(group)]

            c, p, f = _build_zones_for_group(
                pts, centers, col_ids, group_name, scales, dedup,
                reference_pts=ref_norm[:, list(group)] if reference_X is not None else None)
            cols.extend(c)
            pnames.extend(p)
            families.extend(f)

    return cols, pnames, families


def build_zone_adjustments(X, names, zone_X, zone_names, numeric_mask=None):
    """Multiply selected zone indicators by raw features.

    Each product ``zone_i × feature_j`` creates a localized adjustment:
    the feature's value only for samples inside the zone, zero outside.
    """
    names = input_names(names)
    zone_names = input_names(zone_names)
    cols, pnames, families = [], [], []

    if numeric_mask is None:
        feat_idx = list(range(X.shape[1]))
    else:
        feat_idx = list(np.where(numeric_mask)[0])

    for zi in range(zone_X.shape[1]):
        z = zone_X[:, zi]
        if z.std() < 1e-10:
            continue
        for fi in feat_idx:
            adj = z * X[:, fi]
            if adj.std() < 1e-10:
                continue
            cols.append(adj)
            pnames.append(derived_name(f'{zone_names[zi]}*{names[fi]}', zone_names[zi], names[fi]))
            families.append('field_adjust')

    return cols, pnames, families


def neuron_probe_groups(X, y, n_neurons=5, bind_k=1, n_seeds=5,
                        max_per_group=10, n_epochs=2000):
    """Fit a tiny NN and group features by neuron affinity.

    Each feature is assigned to its top ``bind_k`` neurons based on
    average first-layer |weight| across seeds.  Features sharing a
    neuron form a group for zone construction.

    This reduces the combinatorial cost of zone generation on wide
    data: instead of C(d, 3) triples over all features, zones are
    built within each neuron group where C(g, 3) << C(d, 3).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    n_neurons : int
        Number of hidden neurons (= number of groups when bind_k=1).
    bind_k : int
        Each feature binds to its top-k neurons.
        bind_k=1 -> disjoint groups (strictest, most reduction).
        bind_k=2 -> each feature in up to 2 groups (some overlap).
    n_seeds : int
        Number of random seeds to average weights over (stability).
    max_per_group : int
        Cap features per group (top by weight).  Prevents large groups
        from exploding the zone count.
    n_epochs : int
        Training epochs for each probe NN.

    Returns
    -------
    groups : list of dict
        Each dict has:
        - ``'indices'``: list of int (column indices in X)
        - ``'neuron'``: int (neuron index)
        - ``'weights'``: ndarray (per-feature weight for this neuron)
    W_avg : ndarray of shape (n_features, n_neurons)
        Average absolute first-layer weights across seeds.
    """
    try:
        from signalfault.classify._nn import train_nl, compute_norm_stats, normalize
    except ImportError:
        raise ImportError(
            "neuron_cluster_features requires signalfault.classify._nn. "
            "Install signalfault or use potage without this optional function."
        )
    from collections import defaultdict

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sigma = compute_norm_stats(X)
    Xn = normalize(X, mu, sigma)

    # Average |W| across seeds for stability
    d = X.shape[1]
    W_sum = np.zeros((d, n_neurons))
    for seed in range(n_seeds):
        params = train_nl(Xn, y, [n_neurons], n_epochs, seed=seed + 42,
                          lr=0.01, l2=0.01)
        W_sum += np.abs(params['W'][0])
    W_avg = W_sum / n_seeds

    # Assign each feature to its top-k neurons
    neuron_members = defaultdict(list)
    for fi in range(d):
        top_neurons = np.argsort(W_avg[fi])[-bind_k:]
        for ni in top_neurons:
            neuron_members[int(ni)].append(fi)

    # Build output groups, capping per-group size
    groups = []
    for ni in sorted(neuron_members):
        indices = neuron_members[ni]
        ws = W_avg[indices, ni]
        if len(indices) > max_per_group:
            top = np.argsort(ws)[-max_per_group:]
            indices = [indices[i] for i in top]
            ws = ws[top]
        if len(indices) < 2:
            continue
        groups.append({
            'indices': indices,
            'neuron': ni,
            'weights': ws,
        })

    return groups, W_avg


def build_probe_zones(X, names, groups, numeric_mask=None, config=None,
                      dedup=None, reference_X=None):
    """Build zone features within neuron-probe-defined groups.

    Instead of enumerating all C(d,2) or C(d,3) column groups, only
    builds zones within the feature groups identified by
    ``neuron_probe_groups``.  Each group's features are treated as a
    self-contained set for ``build_stage_fields``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    names : list of str
    groups : list of dict from ``neuron_probe_groups``
        Each must have ``'indices'``: list of int.
    numeric_mask : ndarray of bool, optional
        If None, all features within each group treated as numeric.
    config : SoupConfig, optional
    dedup : StructuralDedup, optional
    reference_X : ndarray, optional
        Reference data for fold-safe percentile grids.

    Returns
    -------
    cols : list of ndarray
    pnames : list of str
    families : list of str
    """
    names = input_names(names)
    all_cols, all_names, all_ff = [], [], []
    ref = reference_X if reference_X is not None else X

    for g in groups:
        indices = g['indices']
        if len(indices) < 2:
            continue
        gnames = [names[i] for i in indices]
        X_g = X[:, indices]
        ref_g = ref[:, indices]
        nm = np.ones(len(indices), dtype=bool)
        cols, fnames, ff = build_stage_fields(
            X_g, gnames, nm, config=config, dedup=dedup,
            reference_X=ref_g)
        all_cols.extend(cols)
        all_names.extend(fnames)
        all_ff.extend(ff)

    return all_cols, all_names, all_ff


def build_field_pipeline(X, y, names, numeric_mask=None, config=None,
                         reference_X=None, n_zones=10, n_adjust=15,
                         min_adjust_gain=0.003):
    """Full field pipeline: zones -> select -> adjust with gain filtering.

    1. Generate zone features (circles/spheres from percentile grid)
    2. Greedy-select top ``n_zones`` zones
    3. Build zone x feature adjustment candidates
    4. Greedy-select adjustments, keeping only those with marginal
       R^2 gain above ``min_adjust_gain``

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    names : list of str
    numeric_mask : ndarray of bool, optional
    config : SoupConfig, optional
    reference_X : ndarray, optional
    n_zones : int
        Number of zone features to select in phase 1.
    n_adjust : int
        Max adjustment features to select in phase 2.
    min_adjust_gain : float
        Minimum marginal R^2 for an adjustment to be kept.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_out)
    out_names : list of str
    out_families : list of str
    """
    names = input_names(names)
    from ._library import PrimitiveLibrary
    from ._select import greedy_forward_select

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if numeric_mask is None:
        numeric_mask = np.ones(X.shape[1], dtype=bool)
    else:
        numeric_mask = np.asarray(numeric_mask, dtype=bool)

    # --- Phase 1: generate and select zones ---
    cols, fnames, ff = build_stage_fields(
        X, names, numeric_mask, config=config, reference_X=reference_X)
    if not cols:
        return np.empty((X.shape[0], 0)), [], []

    zone_mask = [i for i, f in enumerate(ff) if f == 'field_zone']
    if not zone_mask:
        return np.empty((X.shape[0], 0)), [], []

    X_zones = np.column_stack([cols[i] for i in zone_mask])
    zone_fn = [fnames[i] for i in zone_mask]

    lib_z = PrimitiveLibrary(X_zones, zone_fn,
                             ['field_zone'] * len(zone_fn))
    h_z = greedy_forward_select(lib_z, y, max_steps=n_zones)
    sel_znames = list(h_z.selected_names())
    sel_zidx = [zone_fn.index(n) for n in sel_znames]
    X_sel_zones = X_zones[:, sel_zidx]

    out_cols = list(X_sel_zones.T)
    out_names = list(sel_znames)
    out_families = ['field_zone'] * len(sel_znames)

    # --- Phase 2: zone x feature adjustments ---
    adj_cols, adj_names, adj_ff = build_zone_adjustments(
        X, names, X_sel_zones, sel_znames, numeric_mask)

    if not adj_cols:
        return np.column_stack(out_cols), out_names, out_families

    X_adj = np.column_stack(adj_cols)
    lib_adj = PrimitiveLibrary(X_adj, adj_names, adj_ff)
    h_adj = greedy_forward_select(lib_adj, y, max_steps=n_adjust)

    for nm, fam, marginal in zip(h_adj.selected_names(),
                                  h_adj.selected_families(),
                                  h_adj.marginal_r2()):
        if marginal < min_adjust_gain:
            continue
        idx = adj_names.index(nm)
        out_cols.append(adj_cols[idx])
        out_names.append(nm)
        out_families.append('field_adjust')

    return np.column_stack(out_cols), out_names, out_families
