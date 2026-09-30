"""Selection routing shared by generated and existing feature sets."""
from numbers import Integral
from ._library import SelectionHistory
from ._select import greedy_forward_select, residual_select, correlation_select, per_class_select
from ._kfold import _kfold_select


def validate_selection(count, method, *, allow_all=False):
    if method not in ('omp', 'greedy', 'correlation'):
        raise ValueError(f"Unknown selection method: {method!r}")
    if count is None and allow_all:
        return
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("select must be a non-negative integer")


def select_features(lib, y, count, method, *, task, folds, seed, y_val=None):
    validate_selection(count, method)
    if task == 'classification':
        if folds is not None and folds > 1 and y_val is None:
            raise NotImplementedError("K-fold evaluation with task='classification' is not yet supported")
        histories = per_class_select(lib, y, method=method, max_features=count, y_val=y_val)
        merged = SelectionHistory('class_union', 'validation' if y_val is not None else 'train')
        seen = set()
        for cls in sorted(histories):
            for step in histories[cls].steps:
                if step['name'] not in seen:
                    seen.add(step['name'])
                    merged.steps.append(dict(step, class_label=cls))
        return merged, histories
    if folds is not None and folds > 1 and y_val is None:
        return _kfold_select(lib.X, lib.names, lib.families, y, count,
                             method, folds, seed), None
    if method == 'greedy':
        history = greedy_forward_select(lib, y, max_steps=count, y_val=y_val)
    elif method == 'omp':
        history = residual_select(lib, y, max_features=count, y_val=y_val)
    else:
        history = correlation_select(lib, y, max_features=count, y_val=y_val)
    return history, None
