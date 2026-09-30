"""FFT frequency discovery and carrier-grid calibration."""

import numpy as np


def calibrate_carrier_freqs(X, y, names, config, n_freqs=4,
                            n_raw=20, n_am=20, exclude_families=None):
    """Calibrate carrier frequency grid via FFT on raw+AM features.

    Builds a quick raw+AM pipeline, FFTs all features against y, and
    returns the top-n frequency bins ranked by aggregate power × breadth.

    Parameters
    ----------
    X : (n, d) array
    y : (n,) array
    names : list of str
    config : SoupConfig
        Used for phase_scale and AM settings.  ``carrier_freqs`` is
        ignored (we're determining it).
    n_freqs : int
        Number of frequencies in the calibrated grid.
    n_raw : int
        Number of raw features to select before AM.
    n_am : int
        Number of AM products to select.
    exclude_families : set or None

    Returns
    -------
    list of float
        Calibrated carrier_freqs grid, sorted ascending.
    """
    # Lazy import to avoid circular dependency
    from ._pipe import Pipeline

    ft = np.full(len(names), 'numeric', dtype='<U11')
    pipe = Pipeline(X, y, names, feature_types=ft, config=config,
                    exclude_families=exclude_families or set(),
                    fold_eval_count=2)
    raw = pipe.raw(select=min(n_raw, len(names)))
    am = pipe.am(raw, select=min(n_am, len(names)))
    fused = pipe.fuse(raw, am)

    X_fused = fused.X
    d = X_fused.shape[1]

    periodicities = fft_freqs(
        X_fused, y, np.arange(d),
        n_per_feature=10,
        phase_scale=config.carrier_phase_scale,
        min_power_ratio=0.05,
        nyquist_fraction=0.4,
    )

    # Collect all raw peaks across features
    all_peaks = []  # (k, power)
    for peaks in periodicities.values():
        for k, power in peaks:
            if k >= 0.25:
                all_peaks.append((k, power))

    if not all_peaks:
        return list(config.carrier_freqs)  # fallback to default

    # Cluster nearby peaks via greedy power-weighted averaging.
    # Sort by k, then merge peaks within merge_radius into clusters.
    merge_radius = 1.5
    all_peaks.sort(key=lambda p: p[0])

    clusters = []  # each: {'k_sum': weighted sum, 'power': total, 'count': n}
    for k, power in all_peaks:
        merged = False
        if clusters:
            c = clusters[-1]
            center = c['k_sum'] / c['power']
            if abs(k - center) <= merge_radius:
                c['k_sum'] += k * power
                c['power'] += power
                c['count'] += 1
                merged = True
        if not merged:
            clusters.append({'k_sum': k * power, 'power': power, 'count': 1})

    # Score each cluster: power * sqrt(count)
    scored = []
    for c in clusters:
        center = c['k_sum'] / c['power']  # power-weighted average k
        score = c['power'] * np.sqrt(c['count'])
        scored.append((center, score, c['power'], c['count']))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Round final k values to 2 decimals for clean feature names
    grid = sorted(set(round(k, 2) for k, _, _, _ in scored[:n_freqs]))

    # Ensure at least one low-frequency entry
    if all(k > 2.0 for k in grid):
        grid = [0.5] + grid[:n_freqs - 1]

    return grid


def fft_freqs(X, y, idx, n_per_feature=5, phase_scale=None,
              min_power_ratio=0.1, max_k=None, nyquist_fraction=0.4):
    """Detect per-feature carrier frequencies via FFT of y-vs-sorted-feature.

    For each feature j, sorts samples by feature j, FFTs the y values along
    that ordering, and returns the top-k frequency peaks.  This finds the
    actual periodicities in the y-feature relationship rather than sweeping
    a fixed grid.

    Parameters
    ----------
    X : (n, d) array
        Feature matrix.
    y : (n,) array
        Target values.
    idx : array-like of int
        Feature column indices to analyze.
    n_per_feature : int
        Maximum number of frequencies to detect per feature.
    phase_scale : float or None
        Phase multiplier for carrier convention (default: pi).
        Returned k values satisfy: sin(k * phase_scale * x_norm).
    min_power_ratio : float
        Minimum spectral power as fraction of strongest peak.
        Peaks below this threshold are discarded.
    max_k : float or None
        Hard maximum carrier k value.  If None, automatically set per
        feature based on ``nyquist_fraction * n_unique_values``.
    nyquist_fraction : float
        Fraction of Nyquist limit to use as per-feature max_k.
        Default 0.4 = stay well below Nyquist to avoid aliasing.
        For a feature with 24 unique values, max cycles = 24*0.4/2 = 4.8
        cycles/unit, which maps to k = 9.6 (with phase_scale=pi).

    Returns
    -------
    dict : feature_index -> list of (k, power) tuples
        Detected frequencies as carrier k values, sorted by power.
        power is normalized (strongest = 1.0).
    """
    _ps = phase_scale if phase_scale is not None else np.pi
    idx = np.asarray(idx)
    n = len(y)

    if n < 8:
        return {}

    result = {}
    for i in idx:
        col = X[:, i]
        finite = ~np.isnan(col)
        if finite.sum() < 8:
            continue
        if np.ptp(col[finite]) < 1e-10:
            continue

        # Per-feature Nyquist: cap based on number of unique values
        n_unique = len(np.unique(col[finite]))
        # Nyquist: max resolvable cycles = n_unique / 2
        # Apply safety fraction to stay well below aliasing
        nyquist_cycles = n_unique * nyquist_fraction / 2.0
        # Convert cycles/unit to k: sin(k * ps * x) has k*ps/(2*pi) cycles
        feature_max_k = nyquist_cycles * 2.0 * np.pi / _ps
        if max_k is not None:
            feature_max_k = min(feature_max_k, max_k)

        if feature_max_k < 0.5:
            continue  # too few unique values for any periodicity

        # Bin feature into ~n_bins equal-count bins, average y per bin.
        # This denoises the y-vs-feature relationship by averaging out
        # the influence of other features (which are noise in this 1D view).
        n_bins = min(max(n_unique // 2, 20), 200)
        order = np.argsort(col)
        y_sorted = y[order].astype(np.float64)

        # Equal-count binning
        bin_size = max(1, n // n_bins)
        y_binned = np.array([
            y_sorted[j * bin_size:min((j + 1) * bin_size, n)].mean()
            for j in range(n_bins)
        ])
        n_eff = len(y_binned)

        # Detrend: remove linear fit so DC and slope don't dominate
        t_axis = np.linspace(0, 1, n_eff)
        slope = (y_binned[-1] - y_binned[0])
        y_detrended = y_binned - (y_binned[0] + slope * t_axis)

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(n_eff)
        y_windowed = y_detrended * window

        # FFT
        spectrum = np.abs(np.fft.rfft(y_windowed))

        # Skip DC (bin 0) and very low frequencies (bin 1 often window artifact)
        spectrum[0] = 0
        if len(spectrum) > 1:
            spectrum[1] = 0

        max_power = spectrum.max()
        if max_power < 1e-10:
            continue

        # FFT bin m corresponds to m cycles over the n sorted samples.
        # The n samples span [0,1] in x_norm space, so bin m = m cycles/unit.
        # sin(k * phase_scale * x_norm) has k * phase_scale / (2*pi) cycles/unit.
        # So k = m * 2*pi / phase_scale.
        fft_freqs_arr = np.arange(len(spectrum))  # bin indices = cycles/unit
        k_values = fft_freqs_arr * (2.0 * np.pi / _ps)

        # Apply per-feature max_k cap (Nyquist-aware)
        valid_mask = k_values <= feature_max_k
        spectrum = spectrum[valid_mask]
        k_values = k_values[valid_mask]

        if len(spectrum) < 2:
            continue

        # Normalize power (max_power already computed above)
        norm_power = spectrum / max_power

        # Find peaks: local maxima above threshold
        peaks = []
        for m in range(1, len(spectrum) - 1):
            if (norm_power[m] > norm_power[m - 1] and
                    norm_power[m] >= norm_power[m + 1] and
                    norm_power[m] >= min_power_ratio and
                    k_values[m] > 0.1):
                peaks.append((float(k_values[m]), float(norm_power[m])))

        # Also check the last bin
        if (len(norm_power) > 2 and
                norm_power[-1] > norm_power[-2] and
                norm_power[-1] >= min_power_ratio and
                k_values[-1] > 0.1):
            peaks.append((float(k_values[-1]), float(norm_power[-1])))

        if not peaks:
            continue

        # Sort by power descending, keep top n
        peaks.sort(key=lambda p: p[1], reverse=True)
        result[int(i)] = peaks[:n_per_feature]

    return result
