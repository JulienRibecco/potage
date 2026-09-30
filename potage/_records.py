"""Feature handles and stage records, independent of pipeline orchestration."""
from collections import Counter
import numpy as np
from ._provenance import input_names

class Stage:
    """Record of a completed pipeline step.

    Stores raw metadata from the step; derived properties are computed
    on access rather than stored redundantly.

    Parameters
    ----------
    name : str
        Display name, e.g. ``'raw'``, ``'am(8d)'``, ``'fuse'``.
    output : FeatureSet
        The feature set produced by this step.
    history : SelectionHistory or None
        Selection trace (None for fuse).
    candidates_generated : int
        Candidate count before filter/dedup.
    candidates_after_filter : int
        Candidate count after filter/dedup, before selection.
    select_budget : int or None
        Requested select count (None for fuse).
    method : str or None
        Selection method used (None for fuse).
    elapsed : float
        Wall time in seconds.
    input_dims : int
        Total input feature count entering this step.
    """

    __slots__ = (
        'name', 'output', 'history',
        'candidates_generated', 'candidates_after_filter',
        'select_budget', 'method', 'elapsed', 'input_dims',
        '_parent_ids', '_step_type', '_per_class_histories',
        '_surviving_names', '_metadata',
    )

    def __init__(self, name, output, history,
                 candidates_generated, candidates_after_filter,
                 select_budget, method, elapsed, input_dims,
                 parent_ids=(), step_type='raw', per_class_histories=None,
                 surviving_names=None, metadata=None):
        self.name = name
        self.output = output
        self.history = history
        self.candidates_generated = candidates_generated
        self.candidates_after_filter = candidates_after_filter
        self.select_budget = select_budget
        self.method = method
        self.elapsed = elapsed
        self.input_dims = input_dims
        self._parent_ids = tuple(parent_ids)
        self._step_type = step_type
        self._per_class_histories = per_class_histories
        self._surviving_names = surviving_names
        self._metadata = metadata

    # -- derived properties --------------------------------------------------

    @property
    def n_selected(self):
        """Number of features in the output."""
        return self.output.X.shape[1]

    @property
    def final_r2(self):
        """Final cumulative R² from selection, or None."""
        if self.history is None or len(self.history) == 0 or self.history.score_kind != 'r2':
            return None
        return self.history.steps[-1]['cumulative_r2']

    @property
    def diagnosis(self):
        """R² curve diagnosis category (lazy import)."""
        if self.history is None or len(self.history) == 0 or self.history.score_kind != 'r2':
            return None
        from ._diagnose import diagnose_curve
        return diagnose_curve(self.history)

    @property
    def family_census(self):
        """``{family: count}`` from the selection history."""
        if self.history is None or len(self.history) == 0:
            return {}
        return dict(Counter(s['family'] for s in self.history.steps))

    @property
    def curve(self):
        """Legacy score values; consult history.score_kind before interpreting."""
        if self.history is None or len(self.history) == 0:
            return []
        return self.history.cumulative_r2()

    # -- display -------------------------------------------------------------

    def __repr__(self):
        r2_str = f', R\u00b2={self.final_r2:.3f}' if self.final_r2 is not None else ''
        return f'Stage({self.name}, {self.n_selected} features{r2_str})'

    def detail(self):
        """Multi-line summary with candidates funnel, R² range, and top features."""
        method_str = self.method or '\u2014'
        lines = [f'{self.name} [{method_str}, {self.elapsed:.2f}s]']

        # Candidates funnel
        lines.append(
            f'  Candidates: {self.candidates_generated:,}'
            f' \u2192 {self.candidates_after_filter:,} after filter'
            f' \u2192 {self.n_selected} selected'
        )

        # R² range + diagnosis
        if self.history is not None and len(self.history) > 0:
            r2_vals = self.curve
            label = "R²" if self.history.score_kind == "r2" else self.history.score_kind
            diag = self.diagnosis
            lines.append(
                f'  {label}: {r2_vals[0]:.3f} \u2192 {r2_vals[-1]:.3f}'
                f'  ({diag})'
            )

            # Top 5 features
            lines.append('  Selection:')
            for i, step in enumerate(self.history.steps[:5]):
                lines.append(
                    f'    {i+1}. {step["name"]:<30s}'
                    f' +{step["marginal_r2"]:.4f}'
                    f'  {label}={step["cumulative_r2"]:.3f}'
                    f'  [{step["family"]}]'
                )
            remaining = len(self.history) - 5
            if remaining > 0:
                lines.append(f'    ... ({remaining} more)')

            # Family census
            census = self.family_census
            if census:
                parts = [f'{fam}({cnt})' for fam, cnt
                         in sorted(census.items(), key=lambda x: -x[1])]
                lines.append(f'  Families: {", ".join(parts)}')
        else:
            lines.append('  (no selection)')

        return '\n'.join(lines)


class FeatureSet:
    """Immutable handle to the output of a pipeline step.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
    names : list of str
    feature_types : ndarray of str, per-column 'numeric'/'bool'/'categorical'
    _X_val : ndarray or None
        Internal — validation data propagated through the pipeline.
        Not part of the public API; used by Pipeline to score
        selection on held-out data.
    """

    __slots__ = ('X', 'names', 'feature_types', '_X_val', '_families')

    def __init__(self, X, names, feature_types, _X_val=None, _families=None):
        object.__setattr__(self, 'X', X)
        object.__setattr__(self, 'names', input_names(names))
        ft = np.asarray(feature_types, dtype='<U11')
        object.__setattr__(self, 'feature_types', ft)
        object.__setattr__(self, '_X_val', _X_val)
        object.__setattr__(self, '_families', list(_families) if _families else None)

    def __setattr__(self, name, value):
        raise AttributeError("FeatureSet is immutable")

    def __reduce__(self):
        return (type(self), (self.X, self.names, self.feature_types,
                            self._X_val, self._families))


    @property
    def source_names(self):
        """Exact raw-column dependencies, one frozenset per output column."""
        return tuple(n.sources for n in self.names)

    def __repr__(self):
        val_str = f", n_val={self._X_val.shape[0]}" if self._X_val is not None else ""
        return f"FeatureSet({self.X.shape[1]} features, {self.X.shape[0]} samples{val_str})"
