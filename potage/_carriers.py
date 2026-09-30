"""Static carrier generation in linear and transformed coordinate spaces."""

from ._feature_utils import _precompute_norm01, _precompute_percentiles
from ._provenance import derived_name, input_names
import numpy as np


_SPACE_LABELS = {
    'linear':      '',
    'log':         'log',
    'log2':        'log2',
    'log10':       'log10',
    'sqrt':        '√',
    'cbrt':        '∛',
    'exp':         'exp',
    'arcsinh':     'arcsinh',
    'logit':       'logit',
    'tanh':        'tanh',
}


_VALID_SPACES = frozenset(_SPACE_LABELS)


def _parse_space(space):
    """Parse space string, returning (canonical_name, extra_param).

    Supports 'pow:p' for arbitrary power-law: sign(x)*|x|^p.
    Example: 'pow:0.33' → ('pow', 0.33).
    All other spaces return (space, None).
    """
    if space.startswith('pow:'):
        try:
            p = float(space[4:])
        except ValueError:
            raise ValueError(f"Invalid pow exponent in {space!r}. Use 'pow:0.5' etc.")
        return 'pow', p
    if space not in _VALID_SPACES:
        raise ValueError(
            f"Unknown feature_space {space!r}. "
            f"Choose from {sorted(_VALID_SPACES)} or 'pow:<exponent>'."
        )
    return space, None


def _space_label(space):
    """Return display label for a feature space string."""
    if space.startswith('pow:'):
        p = space[4:]
        return f'^{p}'
    return _SPACE_LABELS.get(space, space)


def _apply_feature_space(cols, space):
    """Transform columns into the requested coordinate space.

    Parameters
    ----------
    cols : ndarray, shape (n_samples, n_features)
    space : str
        'linear'   — no-op (current behaviour).
        'log'      — signed natural log: sign(x)*log(1+|x|).
        'log2'     — signed base-2 log: sign(x)*log2(1+|x|).
        'log10'    — signed base-10 log: sign(x)*log10(1+|x|). For pH/dB/magnitude.
        'sqrt'     — signed square root: sign(x)*sqrt(|x|).
        'cbrt'     — signed cube root: sign(x)*|x|^(1/3). Weyl/shell-structure space.
        'exp'      — signed exp: sign(x)*expm1(|x|). Inverse of 'log'.
        'arcsinh'  — arcsinh(x). Analytic at 0, handles negatives natively.
                     Preferred over signed-log for income/wealth/intensity data.
        'logit'    — logit(x) = log(x/(1-x)). For proportion/probability features.
                     Clips to (eps, 1-eps) to avoid ±inf.
        'tanh'     — tanh(x). Symmetric compression of heavy tails to (−1,1).
        'pow:<p>'  — sign(x)*|x|^p for arbitrary p > 0.
                     E.g. 'pow:0.33' ≈ cbrt, 'pow:0.5' = sqrt, 'pow:2.0' = signed square.

    Returns the transformed array (copy).
    """
    if space == 'linear':
        return cols
    kind, p = _parse_space(space)
    ax = np.abs(cols)
    sx = np.sign(cols)
    if kind == 'log':
        return sx * np.log1p(ax)
    if kind == 'log2':
        return sx * np.log2(1.0 + ax)
    if kind == 'log10':
        return sx * np.log10(1.0 + ax)
    if kind == 'sqrt':
        return sx * np.sqrt(ax)
    if kind == 'cbrt':
        return sx * np.cbrt(ax)
    if kind == 'exp':
        return sx * np.expm1(ax)
    if kind == 'arcsinh':
        return np.arcsinh(cols)           # handles sign natively, analytic at 0
    if kind == 'logit':
        eps = 1e-6
        x_c = np.clip(cols, eps, 1.0 - eps)
        return np.log(x_c / (1.0 - x_c))
    if kind == 'tanh':
        return np.tanh(cols)
    if kind == 'pow':
        return sx * np.power(ax, p)
    raise ValueError(f"Unhandled feature_space {space!r}")


def build_stage2_static_carriers(X, names, numeric_mask, config,
                                  enabled_families=None, adaptive_freqs=None,
                                  reference_X=None, feature_space='linear'):
    """Static carrier waveforms on single features.

    Parameters
    ----------
    enabled_families : set of str, optional
        Subset of families to generate (e.g. {'carrier_sin', 'carrier_gauss'}).
        None generates all.
    adaptive_freqs : dict, optional
        Per-feature extra frequencies from _adaptive_freqs().
        Maps raw feature index -> list of extra freq values.
    reference_X : array-like, optional
        Reference data (e.g. training set) for norm01 and percentiles.
    feature_space : str, default 'linear'
        Coordinate space applied before [0,1] normalisation:
        'linear' — no transform (current behaviour, backward-compatible).
        'log'    — signed natural log: sign(x)*log(1+|x|).
        'log2'   — signed base-2 log: sign(x)*log2(1+|x|).
        'sqrt'   — signed square root: sign(x)*sqrt(|x|).
        'exp'    — signed exp: sign(x)*expm1(|x|).
        'cbrt'   — signed cube root: sign(x)*|x|^(1/3).  Correct for 3D
                   shell structure (Weyl's law) and particle-count features.
        Step carriers use raw percentile thresholds and are unaffected.
    """
    names = input_names(names)
    cols, pnames, families = [], [], []
    num_idx = np.where(numeric_mask)[0]
    ef = enabled_families
    ref = reference_X

    # Precompute percentiles for step and gauss
    # Step uses raw values (space-independent); gauss uses norm01 space.
    all_pcts = set()
    if ef is None or 'carrier_step' in ef:
        all_pcts.update(config.step_pcts)
    if ef is None or 'carrier_gauss' in ef:
        all_pcts.update(config.gauss_center_pcts)
    pct_table = _precompute_percentiles(X, num_idx, all_pcts, ref_X=ref) if all_pcts else {}

    # Apply feature-space transform before [0,1] normalisation
    X_sp  = _apply_feature_space(X[:, num_idx],
                                  feature_space) if feature_space != 'linear' else X[:, num_idx]
    ref_sp = (_apply_feature_space(ref[:, num_idx], feature_space)
              if (ref is not None and feature_space != 'linear')
              else (ref[:, num_idx] if ref is not None else None))

    # Precompute [0,1] normalized columns (in the chosen space)
    # Reconstruct full arrays expected by _precompute_norm01 (it indexes into them)
    if feature_space != 'linear':
        # Build temporary arrays that _precompute_norm01 can index by num_idx
        X_for_norm   = np.empty_like(X);   X_for_norm[:, num_idx]   = X_sp
        ref_for_norm = (np.empty_like(ref) if ref is not None else None)
        if ref_for_norm is not None:
            ref_for_norm[:, num_idx] = ref_sp
    else:
        X_for_norm   = X
        ref_for_norm = ref

    X_norm, valid = _precompute_norm01(X_for_norm, num_idx, ref_X=ref_for_norm)

    # Precompute percentiles on normalized columns for gauss (batched)
    norm_pct_table = {}
    if ef is None or 'carrier_gauss' in ef:
        valid_iis = [ii for ii in range(len(num_idx)) if valid[ii]]
        if valid_iis and config.gauss_center_pcts:
            pct_list = sorted(config.gauss_center_pcts)
            # Gauss centers from reference norm01 if available (in the chosen space)
            if ref is not None:
                ref_norm, ref_valid = _precompute_norm01(ref_for_norm, num_idx)
                ref_valid_iis = [ii for ii in valid_iis if ref_valid[ii]]
                if ref_valid_iis:
                    ref_norm_cols = ref_norm[:, ref_valid_iis]
                    all_pvals = np.percentile(ref_norm_cols, pct_list, axis=0)
                    for col_j, ii in enumerate(ref_valid_iis):
                        norm_pct_table[ii] = {q: float(all_pvals[qi, col_j])
                                              for qi, q in enumerate(pct_list)}
            else:
                norm_cols = X_norm[:, valid_iis]
                all_pvals = np.percentile(norm_cols, pct_list, axis=0)
                for col_j, ii in enumerate(valid_iis):
                    norm_pct_table[ii] = {q: float(all_pvals[qi, col_j])
                                          for qi, q in enumerate(pct_list)}

    # Name wrapper for non-linear spaces: "fw_mf_ratio" → "log2(fw_mf_ratio)"
    _sp_label = _space_label(feature_space)
    def _fname(raw):
        return derived_name(f'{_sp_label}({raw})', raw) if _sp_label else raw

    for ii, i in enumerate(num_idx):
        if not valid[ii]:
            continue
        xn = X_norm[:, ii]
        fn = _fname(names[i])

        # Sin / Cos
        if ef is None or 'carrier_sin' in ef or 'carrier_cos' in ef:
            # Combine base + adaptive freqs for this feature
            freqs = list(config.carrier_freqs)
            if adaptive_freqs and i in adaptive_freqs:
                freqs = freqs + adaptive_freqs[i]
            _ps = config.carrier_phase_scale
            _pl = config._phase_label
            for k in freqs:
                phase = k * _ps * xn
                if ef is None or 'carrier_sin' in ef:
                    cols.append(np.sin(phase))
                    pnames.append(derived_name(f'sin({k}{_pl}*{fn})', fn))
                    families.append('carrier_sin')
                if ef is None or 'carrier_cos' in ef:
                    cols.append(np.cos(phase))
                    pnames.append(derived_name(f'cos({k}{_pl}*{fn})', fn))
                    families.append('carrier_cos')

        # Triangle
        if ef is None or 'carrier_tri' in ef:
            tri_freqs = list(config.carrier_freqs)
            if adaptive_freqs and i in adaptive_freqs:
                tri_freqs = tri_freqs + adaptive_freqs[i]
            for k in tri_freqs:
                phase = (k * xn) % 1.0
                tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
                cols.append(tri)
                pnames.append(derived_name(f'tri({k}*{fn})', fn))
                families.append('carrier_tri')

        # Step — uses raw percentile thresholds, space-independent
        if ef is None or 'carrier_step' in ef:
            for q in config.step_pcts:
                threshold = pct_table[i][q]
                cols.append((X[:, i] > threshold).astype(np.float64))
                pnames.append(derived_name(f'step({names[i]}>{q}%)', names[i]))   # raw name: threshold is on raw scale
                families.append('carrier_step')

        # Pulse — uses norm01 in chosen space
        if ef is None or 'carrier_pulse' in ef:
            for lo, hi in config.pulse_bands:
                pulse = ((xn >= lo) & (xn <= hi)).astype(np.float64)
                pnames.append(derived_name(f'pulse({fn}:{lo:.0%}-{hi:.0%})', fn))
                cols.append(pulse)
                families.append('carrier_pulse')

        # Gaussian window — uses norm01 in chosen space
        if ef is None or 'carrier_gauss' in ef:
            for q in config.gauss_center_pcts:
                center = norm_pct_table[ii][q]
                for w in config.gauss_widths:
                    gauss = np.exp(-0.5 * ((xn - center) / (w + 1e-12)) ** 2)
                    cols.append(gauss)
                    pnames.append(derived_name(f'gauss({fn},c={q}%,w={w})', fn))
                    families.append('carrier_gauss')

    return cols, pnames, families


def _build_targeted_carriers(X, names, numeric_mask, config, periodicities,
                              reference_X=None):
    """Build carriers only at detected resonant frequencies.

    Like build_stage2_static_carriers but only processes features that
    appear in the periodicities dict, using detected frequencies instead
    of the full config grid.
    """
    names = input_names(names)
    cols, pnames, families = [], [], []
    num_idx = np.where(numeric_mask)[0]
    ref = reference_X

    active_idx = sorted(set(num_idx) & set(periodicities.keys()))
    if not active_idx:
        return cols, pnames, families

    active_arr = np.array(active_idx)
    X_norm, valid = _precompute_norm01(X, active_arr, ref_X=ref)
    all_pcts = set(config.step_pcts) | set(config.gauss_center_pcts)
    pct_table = _precompute_percentiles(X, active_arr, all_pcts, ref_X=ref) if all_pcts else {}

    norm_pct_table = {}
    valid_iis = [ii for ii in range(len(active_idx)) if valid[ii]]
    if valid_iis and config.gauss_center_pcts:
        pct_list = sorted(config.gauss_center_pcts)
        if ref is not None:
            ref_norm, ref_valid = _precompute_norm01(ref, active_arr)
            ref_valid_iis = [ii for ii in valid_iis if ref_valid[ii]]
            if ref_valid_iis:
                ref_norm_cols = ref_norm[:, ref_valid_iis]
                all_pvals = np.percentile(ref_norm_cols, pct_list, axis=0)
                for col_j, ii in enumerate(ref_valid_iis):
                    norm_pct_table[ii] = {q: float(all_pvals[qi, col_j])
                                          for qi, q in enumerate(pct_list)}
        else:
            norm_cols = X_norm[:, valid_iis]
            all_pvals = np.percentile(norm_cols, pct_list, axis=0)
            for col_j, ii in enumerate(valid_iis):
                norm_pct_table[ii] = {q: float(all_pvals[qi, col_j])
                                      for qi, q in enumerate(pct_list)}

    for ii, i in enumerate(active_idx):
        if not valid[ii]:
            continue
        xn = X_norm[:, ii]
        peaks = periodicities[i]

        _ps = config.carrier_phase_scale
        _pl = config._phase_label
        for peak in peaks:
            k = peak[0]
            phase = k * _ps * xn
            sin_v = np.sin(phase)
            cos_v = np.cos(phase)
            cols.append(sin_v)
            pnames.append(derived_name(f'sin({k:.2f}{_pl}*{names[i]})', names[i]))
            families.append('carrier_sin')
            cols.append(cos_v)
            pnames.append(derived_name(f'cos({k:.2f}{_pl}*{names[i]})', names[i]))
            families.append('carrier_cos')
            cols.append(sin_v ** 2)
            pnames.append(derived_name(f'sin\u00b2({k:.2f}{_pl}*{names[i]})', names[i]))
            families.append('carrier_sin2')
            cols.append(cos_v ** 2)
            pnames.append(derived_name(f'cos\u00b2({k:.2f}{_pl}*{names[i]})', names[i]))
            families.append('carrier_cos2')
            tri_phase = (k * xn) % 1.0
            tri = 2.0 * np.abs(2.0 * tri_phase - 1.0) - 1.0
            cols.append(tri)
            pnames.append(derived_name(f'tri({k:.2f}*{names[i]})', names[i]))
            families.append('carrier_tri')

        for q in config.step_pcts:
            threshold = pct_table[i][q]
            cols.append((X[:, i] > threshold).astype(np.float64))
            pnames.append(derived_name(f'step({names[i]}>{q}%)', names[i]))
            families.append('carrier_step')

        for lo, hi in config.pulse_bands:
            pulse = ((xn >= lo) & (xn <= hi)).astype(np.float64)
            cols.append(pulse)
            pnames.append(derived_name(f'pulse({names[i]}:{lo:.0%}-{hi:.0%})', names[i]))
            families.append('carrier_pulse')

        if ii in norm_pct_table:
            for q in config.gauss_center_pcts:
                center = norm_pct_table[ii][q]
                for w in config.gauss_widths:
                    gauss = np.exp(-0.5 * ((xn - center) / (w + 1e-12)) ** 2)
                    cols.append(gauss)
                    pnames.append(derived_name(f'gauss({names[i]},c={q}%,w={w})', names[i]))
                    families.append('carrier_gauss')

    return cols, pnames, families
