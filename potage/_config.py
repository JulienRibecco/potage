"""Configuration shared by feature builders and pipelines."""

import numpy as np


class Config:
    """Controls frequency grids, modulation depths, and feature limits.

    Parameters
    ----------
    carrier_freqs : list of float, or ``'fft'``
        Frequency multipliers for sin/cos/tri carriers.
        If ``'fft'``, the grid is calibrated from data at SoupPipe
        construction time using FFT of y-vs-sorted-feature on raw+AM
        features.  ``calibrate_n_freqs`` controls how many frequencies.
    calibrate_n_freqs : int
        When ``carrier_freqs='fft'``, number of frequencies in the
        calibrated grid.  Default 4 (same size as default static grid).
    fm_depths : list of float
        FM modulation depth values.
    pm_depths : list of float
        PM phase modulation depth values.
    wm_center_pcts : list of int
        Percentiles for window modulation centers.
    wm_shift_scale : float
        Scale factor for WM center shifting.
    wm_width_mods : list of float
        Width modulation factors.
    step_pcts : list of int
        Percentiles for step function thresholds.
    pulse_bands : list of (float, float)
        Normalized bands for pulse features.
    gauss_center_pcts : list of int
        Percentiles for Gaussian window centers.
    gauss_widths : list of float
        Gaussian window widths.
    top_k_modulation : int
        Number of top features to use as carriers/modulators in stages 3-5.
    max_ratios_per_feature : int or None
        Cap on ratio features per denominator. None = unlimited.
    power_scales : list of float
        Exponent multipliers for power features a^(k·b_norm).
        Default [0.5, 1.0, 2.0, 3.0] generates a^(0.5·b), a^b, a^(2b), a^(3b).
    carrier_phase_scale : float
        Phase multiplier for sin/cos/tri carriers and FM/PM modulation.
        Default ``np.pi``: carrier phase = ``k * pi * x_norm``, so
        ``carrier_freqs=[2]`` → one complete cycle over [0,1].
        Set to ``2 * np.pi`` for "number of cycles" semantics:
        ``carrier_freqs=[1, 2, 3]`` → 1, 2, 3 complete cycles.
    """

    def __init__(self, carrier_freqs=None, calibrate_n_freqs=4,
                 fm_depths=None, pm_depths=None,
                 wm_center_pcts=None, wm_shift_scale=1.0, wm_width_mods=None,
                 step_pcts=None, pulse_bands=None, gauss_center_pcts=None,
                 gauss_widths=None, top_k_modulation=15,
                 max_ratios_per_feature=None, power_scales=None,
                 carrier_phase_scale=None,
                 field_pct_levels=None, field_scales=None):
        self.carrier_freqs = carrier_freqs if carrier_freqs is not None else [0.5, 1.0, 2.0, 3.0]
        self.calibrate_n_freqs = calibrate_n_freqs
        self.fm_depths = fm_depths if fm_depths is not None else [-2.0, -1.0, 1.0, 2.0]
        self.pm_depths = pm_depths if pm_depths is not None else [0.5, 1.0, 2.0]
        self.wm_center_pcts = wm_center_pcts if wm_center_pcts is not None else [30, 50, 70]
        self.wm_shift_scale = wm_shift_scale
        self.wm_width_mods = wm_width_mods if wm_width_mods is not None else [0.5, 1.0, 2.0]
        self.step_pcts = step_pcts if step_pcts is not None else [25, 50, 75]
        self.pulse_bands = pulse_bands if pulse_bands is not None else [(0.0, 0.33), (0.33, 0.67), (0.67, 1.0)]
        self.gauss_center_pcts = gauss_center_pcts if gauss_center_pcts is not None else [25, 50, 75]
        self.gauss_widths = gauss_widths if gauss_widths is not None else [0.2, 0.5]
        self.top_k_modulation = top_k_modulation
        self.max_ratios_per_feature = max_ratios_per_feature
        self.power_scales = power_scales if power_scales is not None else [0.5, 1.0, 2.0, 3.0]
        self.carrier_phase_scale = carrier_phase_scale if carrier_phase_scale is not None else np.pi
        self.field_pct_levels = field_pct_levels if field_pct_levels is not None else [20, 40, 60, 80]
        self.field_scales = field_scales if field_scales is not None else [0.1, 0.3, 1.0]
        self.field_max_dims = 2

    @property
    def _phase_label(self):
        """Human-readable label for the carrier phase scale.

        Returns a string that reads correctly after a numeric k value:
        k=2 + label='π' → '2π', k=2 + label='×2π' → '2×2π'.
        """
        s = self.carrier_phase_scale
        if abs(s - np.pi) < 1e-10:
            return 'π'
        if abs(s - 2 * np.pi) < 1e-10:
            return '×2π'
        return f'×{s:.4g}'


SoupConfig = Config  # backward compat alias
