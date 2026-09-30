"""Feature type inference, training-reference statistics, and column accumulation."""

import numpy as np
import warnings


def _infer_feature_types(X, numeric_threshold=20, names=None):
    """Infer per-column feature type: 'numeric', 'bool', or 'categorical'.

    Rules:
      - bool: exactly 2 unique non-NaN values
      - ambiguous (3–threshold unique values): defaults to majority type
        among non-ambiguous columns instead of always 'categorical'
      - numeric: > numeric_threshold unique values

    When ambiguous columns are resolved, logs a summary to help the user
    provide explicit ``feature_types`` for reproducibility.

    Returns ndarray of dtype '<U11' with shape (n_features,).
    """
    n = X.shape[1]
    types = np.empty(n, dtype='<U11')
    ambiguous = []
    for i in range(n):
        vals = X[:, i]
        if np.issubdtype(X.dtype, np.floating):
            vals = vals[~np.isnan(vals)]
        n_unique = len(np.unique(vals))
        if n_unique <= 2:
            types[i] = 'bool'
        elif n_unique <= numeric_threshold:
            types[i] = ''  # ambiguous — resolve below
            ambiguous.append(i)
        else:
            types[i] = 'numeric'

    # Resolve ambiguous columns: default to the majority type among
    # non-ambiguous columns (usually 'numeric' for scientific data).
    if ambiguous:
        non_amb = types[types != '']
        if len(non_amb) > 0:
            unique_types, counts = np.unique(non_amb, return_counts=True)
            majority = unique_types[counts.argmax()]
        else:
            majority = 'numeric'
        for i in ambiguous:
            types[i] = majority

        # Log summary so the user knows inference happened
        amb_names = []
        for i in ambiguous:
            label = names[i] if names and i < len(names) else f'col_{i}'
            vals = X[:, i]
            if np.issubdtype(X.dtype, np.floating):
                vals = vals[~np.isnan(vals)]
            amb_names.append(f'{label}({len(np.unique(vals))}u)')
        warnings.warn(
            f"feature_types not provided: {len(ambiguous)} ambiguous "
            f"column(s) (3–{numeric_threshold} unique values) defaulted "
            f"to '{majority}': {', '.join(amb_names[:10])}"
            + (f' ... and {len(amb_names)-10} more' if len(amb_names) > 10 else '')
            + ". Pass feature_types= explicitly for full control.",
            stacklevel=2)

    return types


def _feature_types_from_numeric_mask(numeric_mask):
    """Convert legacy bool numeric_mask to feature_types array.

    True → 'numeric', False → 'categorical' (conservative default).
    """
    types = np.where(numeric_mask, 'numeric', 'categorical')
    return types.astype('<U11')


def _derive_masks(feature_types):
    """Derive boolean masks from feature_types array.

    Returns (is_numeric, is_bool, is_categorical) — three bool arrays.
    """
    ft = np.asarray(feature_types)
    return (ft == 'numeric', ft == 'bool', ft == 'categorical')


def _precompute_percentiles(X, idx, pcts, ref_X=None):
    """Precompute percentile table for given features and percentile list.

    Parameters
    ----------
    ref_X : array-like, optional
        Reference data for computing percentile thresholds (e.g. training
        set).  When provided, thresholds are computed from ref_X but applied
        as lookup values for X.  When None, thresholds come from X itself.

    Returns dict: feature_index -> {pct: value}
    """
    if not pcts or len(idx) == 0:
        return {}
    pct_list = sorted(pcts)
    source = ref_X[:, idx] if ref_X is not None else X[:, idx]
    # np.percentile with multiple q returns shape (len(q), len(idx))
    all_pvals = np.percentile(source, pct_list, axis=0)
    table = {}
    for ji, i in enumerate(idx):
        table[i] = {q: float(all_pvals[qi, ji]) for qi, q in enumerate(pct_list)}
    return table


def _precompute_norm01(X, idx, ref_X=None):
    """Precompute [0,1]-normalized columns and validity flags.

    Parameters
    ----------
    ref_X : array-like, optional
        Reference data for computing min/max (e.g. training set).  When
        provided, min and max are taken from ref_X but the normalization
        is applied to X.  When None, statistics come from X itself.

    Returns (X_norm, valid) where X_norm is (n_samples, len(idx))
    and valid is boolean array of length len(idx).
    """
    cols = X[:, idx]
    source = ref_X[:, idx] if ref_X is not None else cols
    cmins = source.min(axis=0)
    cmaxs = source.max(axis=0)
    ranges = cmaxs - cmins
    valid = ranges > 1e-10
    ranges_safe = ranges.copy()
    ranges_safe[~valid] = 1.0
    X_norm = (cols - cmins) / ranges_safe
    X_norm[:, ~valid] = 0.0
    return X_norm, valid


class FeatureAccumulator:
    """Pre-allocated output matrix that grows by doubling.

    Replaces cols.append() + column_stack() with direct writes into
    contiguous memory.  add_batch() writes whole matrix blocks with a
    single numpy slice.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    est_features : int
        Initial column capacity (will double when exceeded).
    """

    __slots__ = ('X', 'names', 'families', '_ptr')

    def __init__(self, n_samples, est_features=256):
        self.X = np.empty((n_samples, est_features))
        self.names = []
        self.families = []
        self._ptr = 0

    def add(self, col, name, family):
        """Append a single column."""
        if self._ptr >= self.X.shape[1]:
            self._resize()
        self.X[:, self._ptr] = col
        self.names.append(name)
        self.families.append(family)
        self._ptr += 1

    def add_batch(self, block, batch_names, family):
        """Append a matrix block of k columns at once."""
        k = block.shape[1]
        while self._ptr + k > self.X.shape[1]:
            self._resize()
        self.X[:, self._ptr:self._ptr + k] = block
        self.names.extend(batch_names)
        self.families.extend([family] * k)
        self._ptr += k

    def _resize(self):
        new = np.empty((self.X.shape[0], self.X.shape[1] * 2))
        new[:, :self._ptr] = self.X[:, :self._ptr]
        self.X = new

    def finalize(self):
        """Return (X, names, families) with X trimmed to used columns.

        Returns the data as (cols_list, names_list, families_list) to
        match the builder function return signature.
        """
        if self._ptr == 0:
            return [], [], []
        X_out = self.X[:, :self._ptr]
        # Return as list of columns for API compat with existing builders
        cols = [X_out[:, i] for i in range(self._ptr)]
        return cols, list(self.names), list(self.families)

    def finalize_matrix(self):
        """Return (X_matrix, names, families) without splitting into columns."""
        if self._ptr == 0:
            return None, [], []
        return self.X[:, :self._ptr], list(self.names), list(self.families)
