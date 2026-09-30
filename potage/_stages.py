"""Standalone stage dispatcher and compatibility exports.

Builder implementations live in focused modules. Existing imports from this
module remain supported; internal callers import the implementation directly."""

from ._feature_utils import (
    FeatureAccumulator,
    _derive_masks,
    _feature_types_from_numeric_mask,
    _infer_feature_types,
    _precompute_percentiles,
    _precompute_norm01,
)
from ._library import PrimitiveLibrary
from ._config import SoupConfig, Config
from ._algebra import build_stage0_raw, build_stage1_am
from ._carriers import (
    build_stage2_static_carriers,
    _SPACE_LABELS,
    _VALID_SPACES,
    _parse_space,
    _space_label,
    _apply_feature_space,
    _build_targeted_carriers,
)
from ._fields import (
    build_stage_fields,
    _DEFAULT_PCT_BY_DIM,
    _build_zones_for_group,
    build_zone_adjustments,
    neuron_probe_groups,
    build_probe_zones,
    build_field_pipeline,
)
from ._frequency import calibrate_carrier_freqs, fft_freqs
from ._modulation import (
    _apply_modulation,
    _fm_sin_kernel,
    _pm_sin_kernel,
    _fm_tri_kernel,
    _fm_pulse_kernel,
    _wm_center_kernel,
    _wm_width_kernel,
)
from ._budget import estimate_candidates
import numpy as np


_STAGE_NAMES = {
    'raw': 0, 'am': 1, 'carriers': 2, 'fields': 7, 'probe_fields': 8,
}


def apply_stage(stage, X, names, config=None, y=None,
                numeric_mask=None, feature_types=None,
                include_input=True,
                filter_threshold=0.005, dedup_threshold=0.95,
                input_depths=None,
                max_complexity=None, structural_dedup=False,
                reference_X=None):
    """Apply a single soup stage to arbitrary features.

    Wraps the stage builder functions so you can call any stage on your
    own feature matrix and get back a PrimitiveLibrary.

    Parameters
    ----------
    stage : str
        Which stage: 'raw', 'am', 'carriers', 'fields'.
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.
    names : list of str
        Feature names matching X columns.
    config : SoupConfig, optional
        Stage configuration.  Uses defaults if None.
    y : array-like of shape (n_samples,), optional
        Target vector (unused by current stages, kept for API compat).
    numeric_mask : array-like of bool, optional
        Deprecated — use ``feature_types`` instead.  True for numeric
        features.  Converted to feature_types internally.
    feature_types : array-like of str, optional
        Per-column type: 'numeric', 'bool', or 'categorical'.
        Inferred if neither this nor ``numeric_mask`` is provided.
    include_input : bool
        If True, the returned library includes the input features
        alongside the newly generated ones.
    filter_threshold : float
        Minimum std to keep a feature (0 to disable).
    dedup_threshold : float
        Max correlation for deduplication (0 to disable).
    input_depths : list of int, optional
        Per-feature depth counter from previous stages.  Passthrough
        features (when include_input=True) keep their depth; newly
        generated features get max(input_depths) + 1.  If None,
        input features start at depth 0.
    max_complexity : int, optional
        If set, filter the result to keep only features with
        depth <= max_complexity.
    structural_dedup : bool
        If True, use StructuralDedup to eliminate mathematically
        identical features before computing correlations.  Zero overhead
        when False.
    reference_X : array-like, optional
        Reference data (e.g. training set) for sample-dependent transforms
        (ranks, norm01, percentiles).  When provided, statistics are
        computed from reference_X but applied to X.  When None, statistics
        come from X itself (default behavior).

    Returns
    -------
    PrimitiveLibrary
    """
    X = np.asarray(X, dtype=np.float64)
    names = list(names)
    if reference_X is not None:
        reference_X = np.asarray(reference_X, dtype=np.float64)
    if config is None:
        config = SoupConfig()

    # Resolve feature types (new system) and derive numeric_mask (compat)
    if feature_types is not None:
        feature_types = np.asarray(feature_types)
        is_numeric, is_bool, is_categorical = _derive_masks(feature_types)
        numeric_mask = is_numeric
    elif numeric_mask is not None:
        numeric_mask = np.asarray(numeric_mask, dtype=bool)
        feature_types = _feature_types_from_numeric_mask(numeric_mask)
        is_numeric, is_bool, is_categorical = _derive_masks(feature_types)
    else:
        feature_types = _infer_feature_types(X, names=names)
        is_numeric, is_bool, is_categorical = _derive_masks(feature_types)
        numeric_mask = is_numeric

    # Structural dedup setup
    dedup = None
    if structural_dedup:
        from ._dedup import StructuralDedup
        dedup = StructuralDedup()
        if include_input:
            dedup.seed_raw(X.shape[1])

    # Depth tracking
    n_input = X.shape[1]
    if input_depths is None:
        input_depths = [0] * n_input
    else:
        input_depths = list(input_depths)
    new_depth = max(input_depths) + 1 if input_depths else 1

    if stage not in _STAGE_NAMES:
        raise ValueError(
            f"Unknown stage {stage!r}. Choose from: {list(_STAGE_NAMES)}"
        )
    stage_id = _STAGE_NAMES[stage]

    # Dispatch to builder
    ref = reference_X
    if stage_id == 0:
        cols, pnames, families = build_stage0_raw(X, names, numeric_mask,
                                                  dedup=dedup,
                                                  reference_X=ref,
                                                  feature_types=feature_types)
    elif stage_id == 1:
        cols, pnames, families = build_stage1_am(X, names, numeric_mask, config,
                                                 dedup=dedup, reference_X=ref,
                                                 feature_types=feature_types)
    elif stage_id == 2:
        cols, pnames, families = build_stage2_static_carriers(
            X, names, numeric_mask, config, reference_X=ref)
    elif stage_id == 7:
        cols, pnames, families = build_stage_fields(
            X, names, numeric_mask, config, dedup=dedup,
            reference_X=ref)

    if not cols:
        # No features generated — return input as-is
        return PrimitiveLibrary(
            X, names, ['input'] * len(names),
            complexities=list(input_depths))

    # Assemble output using FeatureAccumulator (avoids column_stack copy)
    n_new = len(pnames)
    n_samples = X.shape[0]
    n_total = (len(names) + n_new) if include_input else n_new
    acc = FeatureAccumulator(n_samples, est_features=n_total)

    if include_input:
        acc.add_batch(X, list(names), 'input')
        # Fix families for input columns (all 'input')
        # add_batch sets all to the same family, which is correct here

    # Add new columns via batch when they're array-like, else one by one
    for i in range(n_new):
        col = cols[i]
        val = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        acc.add(val, pnames[i], families[i])

    X_all, all_names, all_families = acc.finalize_matrix()
    if X_all is None:
        return PrimitiveLibrary(
            X, names, ['input'] * len(names),
            complexities=list(input_depths))

    new_depths = [new_depth] * n_new
    if include_input:
        all_depths = list(input_depths) + new_depths
    else:
        all_depths = new_depths

    lib = PrimitiveLibrary(X_all, all_names, all_families,
                           complexities=all_depths)

    if filter_threshold > 0:
        lib = lib.filter_constant(filter_threshold)
    if dedup_threshold > 0:
        lib = lib.dedup_correlated(dedup_threshold)
    if max_complexity is not None:
        lib = lib.filter_complexity(max_complexity)

    return lib
