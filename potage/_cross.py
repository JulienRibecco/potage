"""Cross-set modulation helpers.

Extracted from _pipe.py — standalone functions for cross-FeatureSet
AM and FM/PM operations with no dependency on SoupPipe internals.
"""

import numpy as np

from ._provenance import derived_name, input_names
from ._feature_utils import _derive_masks, _precompute_norm01


def _apply_cross_modulation(X_c, names_c, types_c, X_m, names_m, types_m,
                            config, kernels,
                            ref_X_c=None, ref_X_m=None):
    """Cross-set modulation: carriers from one set, modulators from another.

    Unlike _apply_modulation which draws both roles from a single pool,
    this restricts carriers to X_c and modulators to X_m — no intra-set pairs.

    Parameters
    ----------
    X_c, names_c, types_c : carrier set data, names, feature types
    X_m, names_m, types_m : modulator set data, names, feature types
    config : SoupConfig
    kernels : list of (kernel_fn, param_grid, family_name)
        Same interface as _apply_modulation kernels.
    ref_X_c : ndarray, optional
        Reference (train) carrier data for norm01 stats.
    ref_X_m : ndarray, optional
        Reference (train) modulator data for norm01 stats.

    Returns
    -------
    cols, pnames, families : lists
    """
    names_c = input_names(names_c)
    names_m = input_names(names_m)
    cols, pnames, families = [], [], []
    is_numeric_c, _, _ = _derive_masks(types_c)

    # Norm01 for carriers and modulators independently
    c_idx = np.arange(X_c.shape[1])
    m_idx = np.arange(X_m.shape[1])
    C_norm, c_valid = _precompute_norm01(X_c, c_idx, ref_X=ref_X_c)
    M_norm, m_valid = _precompute_norm01(X_m, m_idx, ref_X=ref_X_m)

    # Carriers: numeric + valid from carrier set
    carrier_ii = [ii for ii in range(len(c_idx))
                  if is_numeric_c[ii] and c_valid[ii]]
    # Modulators: all valid from mod set
    mod_jj = [jj for jj in range(len(m_idx)) if m_valid[jj]]

    # Cache for carrier stats (used by WM/pulse kernels)
    # Use reference (train) data for stats when available
    cache = {}
    if carrier_ii:
        stat_src = ref_X_c if ref_X_c is not None else X_c
        raw_cols_ref = stat_src[:, [c_idx[ii] for ii in carrier_ii]]
        raw_stds = raw_cols_ref.std(axis=0)
        all_pcts = set()
        all_pcts.update(getattr(config, 'wm_center_pcts', []))
        all_pcts.update(getattr(config, 'gauss_center_pcts', []))
        pct_table = {}
        if all_pcts:
            pct_list = sorted(all_pcts)
            valid_mask = raw_stds > 1e-10
            if valid_mask.any():
                valid_cols = raw_cols_ref[:, valid_mask]
                pct_vals = np.percentile(valid_cols, pct_list, axis=0)
                valid_raw = [carrier_ii[k] for k in range(len(carrier_ii))
                             if valid_mask[k]]
                for col_j, ic in enumerate(valid_raw):
                    pct_table[ic] = {q: float(pct_vals[qi, col_j])
                                     for qi, q in enumerate(pct_list)}
        for k, ic in enumerate(carrier_ii):
            cache[ic] = {
                'std': float(raw_stds[k]),
                'pct': pct_table.get(ic, {}),
            }

    for ii_c in carrier_ii:
        c = C_norm[:, ii_c]
        for jj_m in mod_jj:
            m = M_norm[:, jj_m]

            for kernel_fn, param_grid, family in kernels:
                for params in param_grid:
                    result = kernel_fn(c, m, params, X_c, ii_c, config, cache)
                    if result is None:
                        continue
                    val, suffix = result
                    cols.append(val)
                    pnames.append(derived_name(f'{suffix}_{names_m[jj_m]}\u2192{names_c[ii_c]}', names_m[jj_m], names_c[ii_c]))
                    families.append(family)

    return cols, pnames, families


def _build_cross_am(sets, config, ref_stds=None, reference_X=None):
    """Apply the common AM builder only to pairs from different sets."""
    from ._algebra import build_stage1_am

    X = np.hstack([s.X for s in sets])
    names = [name for s in sets for name in s.names]
    types = np.concatenate([s.feature_types for s in sets])
    groups = [i for i, s in enumerate(sets) for _ in s.names]
    return build_stage1_am(
        X, names, types == 'numeric', config, feature_types=types,
        reference_X=reference_X, pair_groups=groups, ref_stds=ref_stds)
