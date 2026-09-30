"""Raw unary transforms and pairwise algebraic feature generation."""

from ._feature_utils import _derive_masks
from ._provenance import derived_name, input_names
import numpy as np


def build_stage0_raw(X, names, numeric_mask, dedup=None, reference_X=None,
                     feature_types=None):
    """Raw features: identity, squared, log, sqrt, cube, tanh, reciprocal, rank, one-hot.

    Type gating (when ``feature_types`` is provided):
      - **numeric**: all unary transforms (sq, log, sqrt, cube, tanh,
        reciprocal, rank) + raw passthrough.
      - **bool**: raw passthrough only.  Unary transforms on {0,1} produce
        degenerate two-value outputs — skipped.
      - **categorical**: raw passthrough + one-hot expansion.  No unary
        transforms (arithmetic on category codes is meaningless).

    Parameters
    ----------
    dedup : StructuralDedup, optional
        When provided, skip features whose fingerprint was already seen.
    reference_X : array-like, optional
        Reference data (e.g. training set) for sample-dependent transforms.
    feature_types : array-like of str, optional
        Per-column type: 'numeric', 'bool', or 'categorical'.
        When None, falls back to ``numeric_mask`` behavior.
    """
    names = input_names(names)
    cols, pnames, families = [], [], []
    n = X.shape[1]

    # Derive type masks
    if feature_types is not None:
        is_numeric, is_bool, is_categorical = _derive_masks(feature_types)
    else:
        is_numeric = np.asarray(numeric_mask, dtype=bool)
        is_bool = np.zeros(n, dtype=bool)
        is_categorical = ~is_numeric

    def _add(col, name, family, fp):
        if dedup is not None and not dedup.is_new(fp, family):
            return
        cols.append(col)
        pnames.append(name)
        families.append(family)

    # Raw passthrough — all types
    for i in range(n):
        _add(X[:, i], names[i], 'raw', ('raw', (i,)))

    # Unary transforms — numeric only
    num_idx = np.where(is_numeric)[0]
    if len(num_idx) > 0:
        X_num = X[:, num_idx]
        X_abs = np.abs(X_num)
        X_sq = X_num ** 2
        X_log = np.log(X_abs + 1)
        X_sqrt = np.sqrt(X_abs)
        X_cube = X_num ** 3
        X_tanh = np.tanh(X_num)

        for j, i in enumerate(num_idx):
            _add(X_sq[:, j], derived_name(f'{names[i]}^2', names[i]), 'squared', ('sq', (i,)))
        for j, i in enumerate(num_idx):
            _add(X_log[:, j], derived_name(f'log|{names[i]}|', names[i]), 'log', ('log', (i,)))
        for j, i in enumerate(num_idx):
            _add(X_sqrt[:, j], derived_name(f'sqrt|{names[i]}|', names[i]), 'sqrt', ('sqrt', (i,)))
        for j, i in enumerate(num_idx):
            _add(X_cube[:, j], derived_name(f'{names[i]}^3', names[i]), 'cube', ('cube', (i,)))
        for j, i in enumerate(num_idx):
            _add(X_tanh[:, j], derived_name(f'tanh({names[i]})', names[i]), 'tanh', ('tanh', (i,)))

    # 1/(|x|+eps) — reciprocal (numeric only)
    ref = reference_X if reference_X is not None else X
    stds_all = ref.std(axis=0)
    for i in num_idx:
        if stds_all[i] > 1e-10:
            eps = 1e-6 * stds_all[i]
            _add(1.0 / (np.abs(X[:, i]) + eps), derived_name(f'1/|{names[i]}|', names[i]),
                 'reciprocal', ('recip', (i,)))

    # rank(x)/n — numeric only
    for i in num_idx:
        col = X[:, i]
        # Empirical CDF with one convention for fitting, ties and replay.
        sorted_ref = np.sort(ref[:, i])
        ranks = np.searchsorted(sorted_ref, col, side='left').astype(np.float64)
        _add(ranks / len(sorted_ref), derived_name(f'rank({names[i]})', names[i]),
             'rank', ('rank', (i,)))

    # One-hot — categorical only (not bool, not numeric)
    cat_idx = np.where(is_categorical)[0]
    for i in cat_idx:
        vals = X[:, i]
        for v in np.unique(vals[~np.isnan(vals)]):
            _add((vals == v).astype(np.float64), derived_name(f'{names[i]}=={v:g}', names[i]),
                 'onehot', ('onehot', (i, float(v))))

    return cols, pnames, families


def build_stage1_am(X, names, numeric_mask, config=None, dedup=None,
                    feature_types=None, reference_X=None, pair_groups=None, ref_stds=None):
    """Amplitude modulation: pairwise binary operations.

    Type gating (when ``feature_types`` is provided):
      - Products / abs_diff: require at least one numeric operand.
        This keeps bool×numeric (gating) but skips bool×bool and
        categorical×categorical.
      - All other binary ops (ratios, sq_diff, min, max, hmean, normdiff,
        logratio, damped, relu, power): numeric pairs only (unchanged).

    Parameters
    ----------
    dedup : StructuralDedup, optional
        When provided, skip features whose fingerprint was already seen.
    feature_types : array-like of str, optional
        Per-column type: 'numeric', 'bool', or 'categorical'.
    """
    names = input_names(names)
    cols, pnames, families = [], [], []
    n = X.shape[1]
    max_ratios = config.max_ratios_per_feature if config else None

    # Derive type masks for binary op gating
    if feature_types is not None:
        is_numeric, is_bool, is_categorical = _derive_masks(feature_types)
    else:
        is_numeric = np.asarray(numeric_mask, dtype=bool)
        is_bool = np.zeros(n, dtype=bool)
        is_categorical = ~is_numeric

    # Products — require at least one numeric operand
    if n > 1:
        ii, jj = np.triu_indices(n, k=1)
        if pair_groups is not None:
            cross = np.asarray(pair_groups)[ii] != np.asarray(pair_groups)[jj]
            ii, jj = ii[cross], jj[cross]
        # Filter: at least one side must be numeric
        at_least_one_num = is_numeric[ii] | is_numeric[jj]
        ii, jj = ii[at_least_one_num], jj[at_least_one_num]
        if dedup is not None and len(ii) > 0:
            keep = np.array([dedup.is_new(('prod', frozenset({int(ii[k]), int(jj[k])})),
                                          'am_product')
                             for k in range(len(ii))])
            ii, jj = ii[keep], jj[keep]
        if len(ii) > 0:
            prods = X[:, ii] * X[:, jj]
            for k in range(len(ii)):
                cols.append(prods[:, k])
            pnames.extend(derived_name(f'{names[ii[k]]}*{names[jj[k]]}', names[ii[k]], names[jj[k]]) for k in range(len(ii)))
            families.extend(['am_product'] * len(ii))

    # Absolute differences — require at least one numeric operand
    if n > 1:
        ii, jj = np.triu_indices(n, k=1)
        if pair_groups is not None:
            cross = np.asarray(pair_groups)[ii] != np.asarray(pair_groups)[jj]
            ii, jj = ii[cross], jj[cross]
        at_least_one_num = is_numeric[ii] | is_numeric[jj]
        ii, jj = ii[at_least_one_num], jj[at_least_one_num]
        if dedup is not None and len(ii) > 0:
            keep = np.array([dedup.is_new(('absdiff', frozenset({int(ii[k]), int(jj[k])})),
                                          'am_diff')
                             for k in range(len(ii))])
            ii, jj = ii[keep], jj[keep]
        if len(ii) > 0:
            diffs = np.abs(X[:, ii] - X[:, jj])
            for k in range(len(ii)):
                cols.append(diffs[:, k])
            pnames.extend(derived_name(f'|{names[ii[k]]}-{names[jj[k]]}|', names[ii[k]], names[jj[k]]) for k in range(len(ii)))
            families.extend(['am_diff'] * len(ii))

    # Ratios (numeric only, with robustness)
    num_idx = np.where(is_numeric)[0]
    ref = X if reference_X is None else reference_X
    stds = ref.std(axis=0) if ref_stds is None else ref_stds

    for j in num_idx:
        # Skip near-constant denominators
        if stds[j] < 1e-10:
            continue
        eps = 1e-6 * (stds[j] + 1e-10)
        denom = np.abs(X[:, j]) + eps
        ratio_count = 0
        for i in num_idx:
            if i == j or (pair_groups is not None and pair_groups[i] == pair_groups[j]):
                continue
            if max_ratios is not None and ratio_count >= max_ratios:
                break
            fp = ('ratio', (int(i), int(j)))
            if dedup is not None and not dedup.is_new(fp, 'am_ratio'):
                continue
            cols.append(X[:, i] / denom)
            pnames.append(derived_name(f'{names[i]}/{names[j]}', names[i], names[j]))
            families.append('am_ratio')
            ratio_count += 1

    # --- New binary ops (numeric pairs only) ---
    if len(num_idx) > 1:
        nii, njj = np.triu_indices(len(num_idx), k=1)
        if pair_groups is not None:
            groups = np.asarray(pair_groups)[num_idx]
            cross = groups[nii] != groups[njj]
            nii, njj = nii[cross], njj[cross]

        # Apply structural dedup to commutative ops (filter indices before compute)
        if dedup is not None:
            _keep_sq = np.array([dedup.is_new(('sq_diff', frozenset({int(num_idx[nii[k]]), int(num_idx[njj[k]])})),
                                              'am_sq_diff') for k in range(len(nii))])
            _keep_min = np.array([dedup.is_new(('min', frozenset({int(num_idx[nii[k]]), int(num_idx[njj[k]])})),
                                               'am_min') for k in range(len(nii))])
            _keep_max = np.array([dedup.is_new(('max', frozenset({int(num_idx[nii[k]]), int(num_idx[njj[k]])})),
                                               'am_max') for k in range(len(nii))])
            _keep_hm = np.array([dedup.is_new(('hmean', frozenset({int(num_idx[nii[k]]), int(num_idx[njj[k]])})),
                                              'am_hmean') for k in range(len(nii))])
            _keep_nd = np.array([dedup.is_new(('normdiff', frozenset({int(num_idx[nii[k]]), int(num_idx[njj[k]])})),
                                              'am_normdiff') for k in range(len(nii))])
        else:
            _all_true = np.ones(len(nii), dtype=bool)
            _keep_sq = _keep_min = _keep_max = _keep_hm = _keep_nd = _all_true

        Xi = X[:, num_idx[nii]]
        Xj = X[:, num_idx[njj]]
        n_pairs = len(nii)

        def _nname(k):
            return names[num_idx[nii[k]]], names[num_idx[njj[k]]]

        # Squared difference — strain energy proxy
        sq_diffs = (Xi - Xj) ** 2
        for k in range(n_pairs):
            if _keep_sq[k]:
                cols.append(sq_diffs[:, k])
                pnames.append(derived_name(f'({_nname(k)[0]}-{_nname(k)[1]})²', _nname(k)[0], _nname(k)[1]))
                families.append('am_sq_diff')

        # Min / Max — bottleneck / limiting factor
        mins = np.minimum(Xi, Xj)
        maxs = np.maximum(Xi, Xj)
        for k in range(n_pairs):
            if _keep_min[k]:
                cols.append(mins[:, k])
                pnames.append(derived_name(f'min({_nname(k)[0]},{_nname(k)[1]})', _nname(k)[0], _nname(k)[1]))
                families.append('am_min')
            if _keep_max[k]:
                cols.append(maxs[:, k])
                pnames.append(derived_name(f'max({_nname(k)[0]},{_nname(k)[1]})', _nname(k)[0], _nname(k)[1]))
                families.append('am_max')

        # Harmonic mean — 2ab/(a+b), effective rate in series
        sum_ab = Xi + Xj
        sum_safe = np.where(np.abs(sum_ab) > 1e-10, sum_ab, 1.0)
        hmean = np.where(np.abs(sum_ab) > 1e-10,
                         2.0 * Xi * Xj / sum_safe, 0.0)
        for k in range(n_pairs):
            if _keep_hm[k]:
                cols.append(hmean[:, k])
                pnames.append(derived_name(f'hmean({_nname(k)[0]},{_nname(k)[1]})', _nname(k)[0], _nname(k)[1]))
                families.append('am_hmean')

        # Normalized difference — |a-b|/(a+b), scale-invariant mismatch
        abs_sum = np.abs(Xi) + np.abs(Xj)
        abs_sum_safe = np.where(abs_sum > 1e-10, abs_sum, 1.0)
        ndiff = np.where(abs_sum > 1e-10,
                         np.abs(Xi - Xj) / abs_sum_safe, 0.0)
        for k in range(n_pairs):
            if _keep_nd[k]:
                cols.append(ndiff[:, k])
                pnames.append(derived_name(f'|{_nname(k)[0]}-{_nname(k)[1]}|/Σ', _nname(k)[0], _nname(k)[1]))
                families.append('am_normdiff')

        # Log ratio — log(|a|/|b|), relative scaling (batch compute)
        stds_i = stds[num_idx[nii]]
        stds_j = stds[num_idx[njj]]
        lr_valid = (stds_i > 1e-10) & (stds_j > 1e-10)
        if lr_valid.any():
            eps_i_vec = 1e-6 * stds_i[lr_valid]
            eps_j_vec = 1e-6 * stds_j[lr_valid]
            lr_all = (np.log(np.abs(Xi[:, lr_valid]) + eps_i_vec)
                      - np.log(np.abs(Xj[:, lr_valid]) + eps_j_vec))
            lr_indices = np.where(lr_valid)[0]
            for idx_lr, k in enumerate(lr_indices):
                fp = ('logratio', (int(num_idx[nii[k]]), int(num_idx[njj[k]])))
                if dedup is not None and not dedup.is_new(fp, 'am_logratio'):
                    continue
                ni, nj = _nname(k)
                cols.append(lr_all[:, idx_lr])
                pnames.append(derived_name(f'log({ni}/{nj})', ni, nj))
                families.append('am_logratio')

        # Damped ratio — a/(1+|b|), saturation/diminishing returns (batch compute)
        abs_Xj = np.abs(Xj)
        abs_Xi = np.abs(Xi)
        damped_ij = Xi / (1.0 + abs_Xj)
        damped_ji = Xj / (1.0 + abs_Xi)
        for k in range(n_pairs):
            ni, nj = _nname(k)
            ri, rj = int(num_idx[nii[k]]), int(num_idx[njj[k]])
            fp_ij = ('damped', (ri, rj))
            fp_ji = ('damped', (rj, ri))
            if dedup is None or dedup.is_new(fp_ij, 'am_damped'):
                cols.append(damped_ij[:, k])
                pnames.append(derived_name(f'{ni}/(1+|{nj}|)', ni, nj))
                families.append('am_damped')
            if dedup is None or dedup.is_new(fp_ji, 'am_damped'):
                cols.append(damped_ji[:, k])
                pnames.append(derived_name(f'{nj}/(1+|{ni}|)', nj, ni))
                families.append('am_damped')

        # ReLU threshold — max(0, a-b), regime change detector
        relu_ab = np.maximum(0.0, Xi - Xj)
        relu_ba = np.maximum(0.0, Xj - Xi)
        for k in range(n_pairs):
            ri, rj = int(num_idx[nii[k]]), int(num_idx[njj[k]])
            fp_ab = ('relu', (ri, rj))
            fp_ba = ('relu', (rj, ri))
            if dedup is None or dedup.is_new(fp_ab, 'am_relu'):
                cols.append(relu_ab[:, k])
                pnames.append(derived_name(f'relu({_nname(k)[0]}-{_nname(k)[1]})', _nname(k)[0], _nname(k)[1]))
                families.append('am_relu')
            if dedup is None or dedup.is_new(fp_ba, 'am_relu'):
                cols.append(relu_ba[:, k])
                pnames.append(derived_name(f'relu({_nname(k)[1]}-{_nname(k)[0]})', _nname(k)[1], _nname(k)[0]))
                families.append('am_relu')

        # Power variants — a^(k·b_norm) for each exponent scale k
        # Normalize exponents to [0,1] to keep output bounded; different
        # scales k give genuinely different growth curves:
        #   k=0.5: sqrt-like   k=1: linear   k=2: quadratic   k=3: cubic
        power_scales = config.power_scales if config else [0.5, 1.0, 2.0, 3.0]
        if power_scales:
            # Precompute normalized features for exponents (once)
            X_num = X[:, num_idx]
            X_mins = ref[:, num_idx].min(axis=0)
            X_maxs = ref[:, num_idx].max(axis=0)
            X_ranges = X_maxs - X_mins
            X_ranges_safe = np.where(X_ranges > 1e-10, X_ranges, 1.0)
            X_norm = np.clip((X_num - X_mins) / X_ranges_safe, 0.0, 1.0)

            for k in range(n_pairs):
                ni, nj = _nname(k)
                ri, rj = int(num_idx[nii[k]]), int(num_idx[njj[k]])
                ai = Xi[:, k]         # raw base
                bj_norm = X_norm[:, njj[k]]  # normalized exponent
                aj_raw = Xj[:, k]     # raw base (reverse)
                bi_norm = X_norm[:, nii[k]]  # normalized exponent (reverse)

                base_i = np.abs(ai) + 1e-10
                sign_i = np.sign(ai)
                base_j = np.abs(aj_raw) + 1e-10
                sign_j = np.sign(aj_raw)

                for s in power_scales:
                    fp_ij = ('power', (ri, rj), (round(s, 6),))
                    fp_ji = ('power', (rj, ri), (round(s, 6),))

                    if dedup is None or dedup.is_new(fp_ij, 'am_power'):
                        # a^(s·b_norm)  — exponent ∈ [0, s]
                        exp_ij = s * bj_norm
                        val_ij = sign_i * base_i ** exp_ij
                        val_ij = np.nan_to_num(val_ij, nan=0.0, posinf=0.0, neginf=0.0)
                        cols.append(val_ij)
                        if s == 1.0:
                            pnames.append(derived_name(f'{ni}^{nj}', ni, nj))
                        else:
                            pnames.append(derived_name(f'{ni}^({s:g}·{nj})', ni, nj))
                        families.append('am_power')

                    if dedup is None or dedup.is_new(fp_ji, 'am_power'):
                        # b^(s·a_norm)  — reverse direction
                        exp_ji = s * bi_norm
                        val_ji = sign_j * base_j ** exp_ji
                        val_ji = np.nan_to_num(val_ji, nan=0.0, posinf=0.0, neginf=0.0)
                        cols.append(val_ji)
                        if s == 1.0:
                            pnames.append(derived_name(f'{nj}^{ni}', nj, ni))
                        else:
                            pnames.append(derived_name(f'{nj}^({s:g}·{ni})', nj, ni))
                        families.append('am_power')

    return cols, pnames, families
