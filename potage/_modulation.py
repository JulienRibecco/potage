"""Carrier/modulator pairing and FM, PM, and window kernels."""

from ._feature_utils import _precompute_norm01
from ._provenance import derived_name, input_names
import numpy as np


def _apply_modulation(X, names, numeric_mask, config, top_idx, kernels,
                      dedup=None, reference_X=None):
    """Generic modulation builder shared by stages 3-5.

    Parameters
    ----------
    kernels : list of (kernel_fn, param_grid, family_name)
        kernel_fn(carrier, modulator, params, X, i_c, config, cache) -> (array, name_suffix)
        param_grid : list of param dicts
        family_name : str
    dedup : StructuralDedup, optional
        When provided, skip features whose fingerprint was already seen.
    reference_X : array-like, optional
        Reference data (e.g. training set) for norm01, stds, percentiles.

    Returns
    -------
    cols, pnames, families
    """
    names = input_names(names)
    cols, pnames, families = [], [], []
    num_mask_top = numeric_mask[top_idx]
    ref = reference_X

    X_norm, valid = _precompute_norm01(X, top_idx, ref_X=ref)

    carrier_ii = [ii for ii in range(len(top_idx)) if num_mask_top[ii] and valid[ii]]
    mod_ii = [ii for ii in range(len(top_idx)) if valid[ii]]

    # Precompute cache for kernel functions (stds, percentiles)
    cache = {}
    carrier_raw_idx = [top_idx[ii] for ii in carrier_ii]
    if carrier_raw_idx:
        ref_src = ref if ref is not None else X
        raw_cols = ref_src[:, carrier_raw_idx]
        raw_stds = raw_cols.std(axis=0)
        # Collect all percentile values needed
        all_pcts = set()
        all_pcts.update(getattr(config, 'wm_center_pcts', []))
        all_pcts.update(getattr(config, 'gauss_center_pcts', []))
        pct_table = {}
        if all_pcts:
            pct_list = sorted(all_pcts)
            # Filter to carriers with nonzero std
            valid_mask = raw_stds > 1e-10
            if valid_mask.any():
                valid_cols = raw_cols[:, valid_mask]
                pct_vals = np.percentile(valid_cols, pct_list, axis=0)
                valid_raw_idx = [carrier_raw_idx[k] for k in range(len(carrier_raw_idx)) if valid_mask[k]]
                for col_j, i_c in enumerate(valid_raw_idx):
                    pct_table[i_c] = {q: float(pct_vals[qi, col_j])
                                      for qi, q in enumerate(pct_list)}
        for k, i_c in enumerate(carrier_raw_idx):
            cache[i_c] = {
                'std': float(raw_stds[k]),
                'pct': pct_table.get(i_c, {}),
            }

    for ii_c in carrier_ii:
        c = X_norm[:, ii_c]
        i_c = top_idx[ii_c]
        for ii_m in mod_ii:
            if ii_m == ii_c:
                continue
            m = X_norm[:, ii_m]
            i_m = top_idx[ii_m]

            for kernel_fn, param_grid, family in kernels:
                for params in param_grid:
                    # Build fingerprint for modulation: (family, (mod, carrier), param_tuple)
                    if dedup is not None:
                        param_key = tuple(round(v, 6) if isinstance(v, float)
                                          else v for v in sorted(params.items()))
                        fp = (family, (int(i_m), int(i_c)), param_key)
                        if not dedup.is_new(fp, family):
                            continue

                    result = kernel_fn(c, m, params, X, i_c, config, cache)
                    if result is None:
                        continue
                    val, suffix = result
                    cols.append(val)
                    pnames.append(derived_name(f'{suffix}_{names[i_m]}\u2192{names[i_c]}', names[i_m], names[i_c]))
                    families.append(family)

    return cols, pnames, families


def _fm_sin_kernel(c, m, params, X, i_c, config, cache):
    k, d = params['k'], params['d']
    _ps = config.carrier_phase_scale
    k_eff = k + d * m
    return np.sin(k_eff * _ps * c), f'FM_sin_k{k}_d{d}'


def _pm_sin_kernel(c, m, params, X, i_c, config, cache):
    k, p = params['k'], params['p']
    _ps = config.carrier_phase_scale
    return np.sin(k * _ps * c + p * _ps * m), f'PM_sin_k{k}_p{p}'


def _fm_tri_kernel(c, m, params, X, i_c, config, cache):
    k, d = params['k'], params['d']
    k_eff = k + d * m
    phase = (k_eff * c) % 1.0
    tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    return tri, f'FM_tri_k{k}_d{d}'


def _fm_pulse_kernel(c, m, params, X, i_c, config, cache):
    q, w = params['q'], params['w']
    # Use cached percentile if available
    if cache and i_c in cache and q in cache[i_c].get('pct', {}):
        center = cache[i_c]['pct'][q]
    else:
        center = np.percentile(c, q)
    eff_center = center + config.wm_shift_scale * (m - 0.5)
    gauss = np.exp(-0.5 * ((c - eff_center) / (w + 1e-12)) ** 2)
    return gauss, f'FM_pulse_c{q}_w{w}'


def _wm_center_kernel(c_norm, m, params, X, i_c, config, cache):
    """Window modulation with center shift. Uses cached raw carrier stats."""
    q = params['q']
    c_raw = X[:, i_c]
    if cache and i_c in cache:
        c_std = cache[i_c]['std']
        pct = cache[i_c].get('pct', {})
        base_center = pct.get(q)
        if base_center is None:
            base_center = np.percentile(c_raw, q)
    else:
        c_std = np.std(c_raw)
        base_center = np.percentile(c_raw, q)
    if c_std < 1e-10:
        return None
    width = 0.3 * c_std
    eff_center = base_center + c_std * (m - 0.5)
    wm = np.exp(-0.5 * ((c_raw - eff_center) / (width + 1e-12)) ** 2)
    return wm, f'WM_center_q{q}'


def _wm_width_kernel(c_norm, m, params, X, i_c, config, cache):
    """Window modulation with width modulation. Uses cached raw carrier stats."""
    q, wmod = params['q'], params['wmod']
    c_raw = X[:, i_c]
    if cache and i_c in cache:
        c_std = cache[i_c]['std']
        pct = cache[i_c].get('pct', {})
        base_center = pct.get(q)
        if base_center is None:
            base_center = np.percentile(c_raw, q)
    else:
        c_std = np.std(c_raw)
        base_center = np.percentile(c_raw, q)
    if c_std < 1e-10:
        return None
    base_width = 0.3 * c_std
    eff_width = base_width * (1.0 + wmod * (m - 0.5))
    eff_width = np.maximum(eff_width, 1e-12)
    wm = np.exp(-0.5 * ((c_raw - base_center) / eff_width) ** 2)
    return wm, f'WM_width_q{q}_m{wmod}'
