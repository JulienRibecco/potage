#!/usr/bin/env python3
"""Paired AM-only defect audit on concrete and optional metallic-glass data.

The legacy arm changes only AM reference statistics to batch-local statistics.
All rank, selection, deduplication, and pipeline fixes remain enabled in both
arms. Never use the legacy context for production predictions. It patches
process-global builder bindings and is intentionally single-threaded.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from potage import Pipeline, Config
from potage import _algebra
from examples.concrete_strength import load_data, recipe_groups, ALPHAS

AFFECTED = {'am_ratio', 'am_logratio', 'am_power'}


@contextmanager
def am_mode(mode):
    """Emulate the old AM statistics without reverting unrelated fixes.

    Cross-set callers already supplied training standard deviations before
    the fix, so ref_stds is preserved. Only reference_X is discarded. For
    powers, batch-local extrema put all exponents within [0,1], as before.
    """
    if mode == 'fixed':
        yield
        return
    if mode != 'legacy_batch':
        raise ValueError(mode)
    original = _algebra.build_stage1_am

    def batch_local(*args, **kwargs):
        kwargs['reference_X'] = None
        return original(*args, **kwargs)

    bindings = [(module, name) for module_name, module in list(sys.modules.items())
                if module_name.startswith('potage.') and module is not None
                for name, value in list(vars(module).items()) if value is original]
    with ExitStack() as stack:
        for module, name in bindings:
            stack.enter_context(patch.object(module, name, batch_local))
        yield


def grouped_split(X, y, groups, seed):
    outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    development, test = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
    train, validation = next(inner.split(X[development], y[development], groups[development]))
    train, validation = development[train], development[validation]
    sets = [set(groups[idx]) for idx in (train, validation, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return train, validation, test


def scores(y, prediction):
    return {'r2': float(r2_score(y, prediction)),
            'mae': float(mean_absolute_error(y, prediction))}


def readout(X, y):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)


def fit_recipe(X, y, names, split, recipe, mode, control=False):
    train, validation, test = split
    exclude = set() if recipe == 'concrete_am' else {'reciprocal', 'rank'}
    if control:
        exclude |= AFFECTED
    p = Pipeline(X[train], y[train], names, feature_types=['numeric'] * len(names),
                 X_val=X[validation], y_val=y[validation], max_memory=None,
                 max_candidates=100000, config=Config(max_ratios_per_feature=2)
                 if recipe != 'concrete_am' else Config(), exclude_families=exclude)
    with am_mode(mode):
        raw = p.raw(select=None if recipe == 'concrete_am' else 10, method='greedy')
        if recipe == 'glass_cross_am':
            # Match the two overlapping ranked raw pools in the historical
            # metallic-glass soup probe. These are NOT disjoint feature views.
            other = p.raw(select=8, method='greedy')
            am = p.am(raw, other, select=None)
        else:
            am = p.am(raw, select=None)
        final = p.select(p.fuse(raw, am), select=20, method='greedy')
        model = readout(final.X, y[train])
        val_prediction = model.predict(p.transform(X[validation]))
        test_prediction = model.predict(p.transform(X[test]))
    return p, model, final, val_prediction, test_prediction


def replay_diagnostic(p, model, X, full_prediction, mode):
    n = min(16, len(X))
    with warnings.catch_warnings(record=True) as caught, am_mode(mode):
        warnings.simplefilter('always')
        single = np.concatenate([model.predict(p.transform(row[None])) for row in X[:n]])
        small_batch = model.predict(p.transform(X[:n]))
    return {'rows_checked': n,
            'single_row_max_abs_prediction_change': float(np.max(np.abs(single - full_prediction[:n]))),
            'small_batch_max_abs_prediction_change': float(np.max(np.abs(small_batch - full_prediction[:n]))),
            'warnings': sorted({str(w.message) for w in caught})}


def audit_case(X, y, names, groups, recipe, seed):
    split = grouped_split(X, y, groups, seed)
    train, validation, test = split
    baseline = readout(X[train], y[train])
    row = {'recipe': recipe, 'seed': seed,
           'split_rows': dict(zip(('discovery', 'validation', 'test'), map(len, split))),
           'split_indices_sha256': [hashlib.sha256(np.asarray(idx, dtype='<i8').tobytes()).hexdigest() for idx in split],
           'raw_ridge_test': scores(y[test], baseline.predict(X[test]))}
    fitted = {}
    for mode in ('legacy_batch', 'fixed'):
        p, model, final, vp, tp = fit_recipe(X, y, names, split, recipe, mode)
        row[mode] = {'validation': scores(y[validation], vp), 'test': scores(y[test], tp),
                     'features': list(final.names), 'families': final._families,
                     'ridge_alpha': float(model[-1].alpha_),
                     'batch_diagnostic': replay_diagnostic(p, model, X[test], tp, mode)}
        fitted[mode] = (p, model, final, vp, tp)
    # Freeze each fitted model to separate replay effects from selection changes.
    for fit_mode, (p, model, final, vp, tp) in fitted.items():
        other_mode = 'fixed' if fit_mode == 'legacy_batch' else 'legacy_batch'
        with am_mode(other_mode):
            other_prediction = model.predict(p.transform(X[test]))
        row[fit_mode]['test_with_other_replay'] = scores(y[test], other_prediction)
    left, right = set(row['legacy_batch']['features']), set(row['fixed']['features'])
    row['selected_feature_jaccard'] = len(left & right) / len(left | right) if left | right else 1.0
    row['delta_fixed_minus_legacy'] = {metric: row['fixed']['test'][metric] - row['legacy_batch']['test'][metric]
                                      for metric in ('r2', 'mae')}
    # Negative control: remove all affected families before dedup/selection.
    controls = [fit_recipe(X, y, names, split, recipe, mode, control=True)
                for mode in ('legacy_batch', 'fixed')]
    row['unaffected_family_control'] = {
        'same_features': controls[0][2].names == controls[1][2].names,
        'max_abs_prediction_change': float(np.max(np.abs(controls[0][4] - controls[1][4])))}
    assert row['unaffected_family_control']['same_features']
    assert row['unaffected_family_control']['max_abs_prediction_change'] < 1e-8
    assert row['fixed']['batch_diagnostic']['single_row_max_abs_prediction_change'] < 1e-6
    print(f"{recipe:16s} seed={seed}: test R2 {row['legacy_batch']['test']['r2']:.4f} -> {row['fixed']['test']['r2']:.4f}; "
          f"MAE {row['legacy_batch']['test']['mae']:.3f} -> {row['fixed']['test']['mae']:.3f}", flush=True)
    return row


def load_glass(path):
    with path.open() as handle:
        reader = csv.DictReader(handle)
        names = [n for n in reader.fieldnames if n not in {'Tg', 'alloy', 'base_element', 'mag_fraction'}]
        rows = list(reader)
    X = np.array([[float(row[n]) for n in names] for row in rows])
    y = np.array([float(row['Tg']) for row in rows])
    assert np.isfinite(X).all() and np.isfinite(y).all()
    # Identical descriptor rows are grouped, including repeated measurements.
    _, groups = np.unique(X, axis=0, return_inverse=True)
    return X, y, names, groups


def save_plot(result, path):
    """Plot paired test MAE for every split; units stay in separate panels."""
    import matplotlib.pyplot as plt
    recipes = [('concrete_am', 'Concrete · AM', 'MPa'),
               ('glass_am', 'Metallic glass · AM', 'K'),
               ('glass_cross_am', 'Metallic glass · cross-AM', 'K')]
    recipes = [item for item in recipes if any(r['recipe'] == item[0] for r in result['runs'])]
    fig, axes = plt.subplots(1, len(recipes), figsize=(4.2 * len(recipes), 4.4), squeeze=False)
    colors = ['#146C94', '#DA7843', '#72833C', '#945DA8']
    for ax, (recipe, title, unit) in zip(axes[0], recipes):
        rows = [r for r in result['runs'] if r['recipe'] == recipe]
        for color, row in zip(colors * len(rows), rows):
            values = [row[m]['test']['mae'] for m in ('legacy_batch', 'fixed')]
            ax.plot([0, 1], values, '-o', color=color, linewidth=2, label=f"Split seed {row['seed']}")
        base = np.mean([r['raw_ridge_test']['mae'] for r in rows])
        ax.axhline(base, color='#777777', linestyle='--', linewidth=1.2, label='Raw ridge mean')
        ax.set_xticks([0, 1], ['Batch-local AM', 'Fixed AM'])
        ax.set_xlim(-.2, 1.2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Test MAE ({unit}) · lower is better')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('AM reference-statistics audit · paired grouped splits', fontsize=14)
    fig.tight_layout(rect=(0, .09, 1, .93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concrete-data-home', default='.cache')
    parser.add_argument('--metallic-glass-csv', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 19, 97])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    from importlib.metadata import version
    started = time.perf_counter()
    X, y, names = load_data(args.concrete_data_home)
    datasets = [('concrete', X, y, names, recipe_groups(X), ['concrete_am'], 'MPa')]
    if args.metallic_glass_csv:
        Xg, yg, ng, gg = load_glass(args.metallic_glass_csv)
        datasets.append(('metallic_glass', Xg, yg, ng, gg, ['glass_am', 'glass_cross_am'], 'K'))
    result = {'purpose': 'AM-only paired defect audit; not an independent benchmark confirmation',
              'modes': {'legacy_batch': 'batch-local AM statistics; all other current fixes retained',
                        'fixed': 'training-reference AM statistics and bounded power exponents'},
              'metallic_glass_scope': 'Focused raw-to-AM ridge audit on the research dataset; not a rerun of the historical neural cascade or MEGNet augmentation',
              'environment': {'python': platform.python_version(), **{n: version(n) for n in ('numpy','scipy','scikit-learn')}},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [*sorted((ROOT/'potage').glob('*.py')), Path(__file__).resolve(),
                                          ROOT/'examples/concrete_strength.py']},
              'datasets': {}, 'runs': []}
    for label, X, y, names, groups, recipes, unit in datasets:
        result['datasets'][label] = {'rows': len(y), 'columns': len(names), 'groups': len(np.unique(groups)),
                                     'target_unit': unit, 'data_sha256': hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()}
        if label == 'metallic_glass':
            result['datasets'][label]['source_file_sha256'] = hashlib.sha256(args.metallic_glass_csv.read_bytes()).hexdigest()
        for recipe in recipes:
            for seed in args.seeds:
                row = audit_case(X, y, names, groups, recipe, seed)
                row['dataset'] = label
                result['runs'].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result['elapsed_seconds'] = time.perf_counter() - started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.plot:
        save_plot(result, args.plot)
    print(f'Wrote {args.output} ({result["elapsed_seconds"]:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
