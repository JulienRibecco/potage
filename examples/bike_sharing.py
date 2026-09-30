#!/usr/bin/env python3
"""Chronological bike-sharing benchmark for current Potage feature recipes.

Default runs score validation only. Add --evaluate-test after fixing the recipe
list/configuration to report the untouched final quarter. No demand lags or
rental-count components are inputs; weather is observed, not forecast weather.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import urllib.request
import zipfile

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from potage import Config, Pipeline

UCI_URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
ARCHIVE_SHA256 = 'b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401'
NAMES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
         'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
TYPES = ['categorical', 'bool', 'numeric', 'numeric', 'bool', 'numeric', 'bool',
         'categorical', 'numeric', 'numeric', 'numeric', 'numeric']
ALPHAS = np.logspace(-3, 4, 15)
RECIPES = ['unary', 'carriers', 'fft_carriers', 'carrier_am', 'carrier_fm',
           'carrier_wm', 'fields', 'carrier_spaces']


def parse_hourly(content):
    rows = sorted(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))),
                  key=lambda row: (row['dteday'], int(row['hr'])))
    dates = np.array([row['dteday'] for row in rows])
    stamps = [row['dteday'] + ':' + row['hr'] for row in rows]
    if len(set(stamps)) != len(stamps):
        raise ValueError('Expected distinct dated hourly observations')
    X = np.array([[float(row[name]) for name in NAMES] for row in rows])
    y = np.array([float(row['cnt']) for row in rows])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Hourly data contains nonfinite values')
    return X, y, dates


def load_data(data_home):
    path = Path(data_home) / 'bike-sharing.zip'
    if not path.exists():
        request = urllib.request.Request(UCI_URL, headers={'User-Agent': 'Potage-example/0.1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError('UCI archive changed: verify its contents before updating the pinned hash')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Cached bike-sharing archive does not match the pinned UCI archive')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        hourly = archive.read('hour.csv')
    return (*parse_hourly(hourly), hashlib.sha256(hourly).hexdigest())


def split_indices(dates):
    train = np.flatnonzero(dates < '2012-07-01')
    validation = np.flatnonzero((dates >= '2012-07-01') & (dates < '2012-10-01'))
    test = np.flatnonzero(dates >= '2012-10-01')
    if not all(len(indices) for indices in (train, validation, test)):
        raise ValueError('All three chronological windows must contain rows')
    assert max(dates[train]) < min(dates[validation])
    assert max(dates[validation]) < min(dates[test])
    return train, validation, test


def cyclic_matrix(X, interactions=False):
    # True calendar periods, independent of observed training minima/maxima.
    extra = []
    for name, period, harmonics in [('hr', 24, 4), ('mnth', 12, 2), ('weekday', 7, 2)]:
        column = X[:, NAMES.index(name)]
        for harmonic in range(1, harmonics + 1):
            phase = 2 * np.pi * harmonic * column / period
            extra.extend([np.sin(phase), np.cos(phase)])
    if interactions:
        hour_terms = extra[:8]
        for context in ['workingday', 'temp']:
            extra.extend(term * X[:, NAMES.index(context)] for term in hour_terms)
    return np.column_stack(extra)


def manual_encoder(X_train, *, cyclic=False, interactions=False):
    categorical = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    if not cyclic:
        categorical += ['mnth', 'hr', 'weekday']
    indices = [NAMES.index(name) for name in categorical]
    encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train[:, indices])
    weather = [NAMES.index(name) for name in ['temp', 'atemp', 'hum', 'windspeed']]

    def transform(X):
        parts = [encoder.transform(X[:, indices]).toarray(), X[:, weather]]
        if cyclic:
            parts.append(cyclic_matrix(X, interactions=interactions))
        return np.column_stack(parts)
    return transform


def score(y, prediction):
    return {'r2': float(r2_score(y, prediction)),
            'mae_rentals': float(mean_absolute_error(y, prediction)),
            'rmse_rentals': float(np.sqrt(mean_squared_error(y, prediction)))}


def fit_ridge(X, y, X_val, y_val):
    best, best_error, best_alpha = None, float('inf'), None
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(X, y)
        error = mean_squared_error(y_val, model.predict(X_val))
        if error < best_error:
            best, best_error, best_alpha = model, error, float(alpha)
    return best, best_alpha


def build_recipe(X, y, X_val, y_val, recipe, select=24):
    config = Config(carrier_freqs=[1, 2, 3, 4], carrier_phase_scale=2 * np.pi,
                    fm_depths=[-1, 1], pm_depths=[0.5, 1], top_k_modulation=8,
                    max_ratios_per_feature=2, power_scales=[0.5, 1, 2],
                    field_pct_levels=[25, 50, 75], field_scales=[0.15, 0.4])
    pipe = Pipeline(X, y, NAMES, feature_types=TYPES, config=config,
                    X_val=X_val, y_val=y_val, fold_eval_count=None,
                    max_candidates=40_000, max_memory=2_000_000_000,
                    exclude_families={'reciprocal'})
    raw = pipe.raw(select=select if recipe == 'unary' else 16, method='greedy')
    if recipe == 'unary':
        return pipe, raw
    if recipe == 'fields':
        narrow = pipe.select(raw, select=8, method='greedy')
        fields = pipe.fields(narrow, select=select, method='greedy')
        final = pipe.select(pipe.fuse(raw, fields), select=select, method='greedy')
        return pipe, final
    if recipe == 'carrier_spaces':
        final, _ = pipe.discover_carrier_space(raw, select=select, method='greedy',
                                              spaces=['linear', 'log', 'tanh'])
        return pipe, final
    if recipe == 'fft_carriers':
        carriers = pipe.fft_carriers(raw, select=select, method='greedy', n_per_feature=4)
    else:
        carriers = pipe.carriers(raw, select=select, method='greedy')
    pool = [raw, carriers]
    if recipe == 'carrier_am':
        narrow = pipe.select(carriers, select=12, method='greedy')
        pool.append(pipe.am(narrow, select=select, method='greedy'))
    elif recipe in ('carrier_fm', 'carrier_wm'):
        narrow = pipe.select(carriers, select=8, method='greedy')
        modulators = pipe.select(raw, select=8, method='greedy')
        stage = pipe.fm if recipe == 'carrier_fm' else pipe.wm
        pool.append(stage(narrow, modulators, select=select, method='greedy'))
    final = pipe.select(pipe.fuse(*pool), select=select, method='greedy')
    return pipe, final


def run_benchmark(data_home, recipes, select=24, evaluate_test=False):
    from importlib.metadata import version
    X, y, dates, hour_hash = load_data(data_home)
    split = split_indices(dates)
    train, validation, test = split
    Xt, yt, Xv, yv = X[train], y[train], X[validation], y[validation]
    result = {
        'dataset': {'doi': 'https://doi.org/10.24432/C5W894', 'source_url': UCI_URL,
                    'archive_sha256': ARCHIVE_SHA256, 'hour_csv_sha256': hour_hash,
                    'rows': len(y), 'features': NAMES, 'excluded_columns': ['instant', 'dteday', 'casual', 'registered'],
                    'dates_used_for_splitting_only': True, 'deduplicated': False},
        'protocol': {'selection_metric': 'validation RMSE', 'test_evaluated': evaluate_test,
                     'refit_on_validation': False, 'observed_weather': True,
                     'alphas': ALPHAS.tolist(), 'recipes': recipes, 'select': select,
                     'splits': {name: {'rows': len(idx), 'from': str(min(dates[idx])), 'through': str(max(dates[idx])),
                                       'indices_sha256': hashlib.sha256(np.asarray(idx, dtype='<i8').tobytes()).hexdigest()}
                                for name, idx in zip(['train', 'validation', 'test'], split)}},
        'environment': {'python': platform.python_version(), **{name: version(name) for name in ['numpy', 'scipy', 'scikit-learn']}},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((ROOT/'potage').glob('*.py')), Path(__file__).resolve()]},
        'models': {},
    }
    fitted = {}

    def add_ridge(name, transform, names=None, stages=None):
        started = perf_counter()
        Zt, Zv = transform(Xt), transform(Xv)
        model, alpha = fit_ridge(Zt, yt, Zv, yv)
        result['models'][name] = {'validation': score(yv, model.predict(Zv)),
                                  'features': Zt.shape[1], 'ridge_alpha': alpha,
                                  'readout_fit_seconds': perf_counter()-started}
        if names is not None:
            result['models'][name]['selected_names'] = list(map(str, names))
        if stages is not None:
            result['models'][name]['stages'] = stages
        fitted[name] = (model, transform)
        print(name, 'validation', result['models'][name]['validation'], flush=True)

    add_ridge('raw_ridge', lambda data: data)
    add_ridge('calendar_onehot_ridge', manual_encoder(Xt))
    add_ridge('cyclical_ridge', manual_encoder(Xt, cyclic=True))
    add_ridge('cyclical_interactions_ridge', manual_encoder(Xt, cyclic=True, interactions=True))
    started = perf_counter()
    tree = HistGradientBoostingRegressor(max_iter=300, l2_regularization=1,
                                         early_stopping=False, random_state=42).fit(Xt, yt)
    result['models']['hist_gradient_boosting'] = {'validation': score(yv, tree.predict(Xv)),
                                                'features': X.shape[1], 'fit_seconds': perf_counter()-started}
    fitted['hist_gradient_boosting'] = (tree, lambda data: data)
    print('hist_gradient_boosting validation', result['models']['hist_gradient_boosting']['validation'], flush=True)

    for recipe in recipes:
        started = perf_counter()
        print('Building', recipe, flush=True)
        pipe, final = build_recipe(Xt, yt, Xv, yv, recipe, select)
        discovery_seconds = perf_counter()-started
        stages = [{'name': stage.name, 'generated': stage.candidates_generated,
                   'after_filter': stage.candidates_after_filter, 'selected': stage.n_selected}
                  for stage in pipe.stages]
        # Confirm that replay is the fitted feature matrix before fitting the readout.
        np.testing.assert_allclose(pipe.transform(Xt), final.X, rtol=1e-10, atol=1e-10)
        name = 'potage_' + recipe
        add_ridge(name, pipe.transform, final.names, stages)
        model, transform = fitted[name]
        batch = model.predict(transform(Xv[:24]))
        singles = np.array([model.predict(transform(Xv[[i]]))[0] for i in range(min(24, len(Xv)))])
        np.testing.assert_allclose(batch, singles, rtol=1e-10, atol=1e-8)
        result['models'][name]['validation_batch_max_abs_difference'] = float(np.max(np.abs(batch-singles)))
        result['models'][name]['discovery_seconds'] = discovery_seconds
        result['models'][name]['source_names'] = [sorted(sources) for sources in final.source_names]
        print('Discovery seconds:', round(discovery_seconds, 2), flush=True)

    # Freeze both choices BEFORE any test predictions or test scores are computed.
    potage_names = [name for name in fitted if name.startswith('potage_')]
    criterion = lambda name: result['models'][name]['validation']['rmse_rentals']
    result['chosen_potage_by_validation'] = min(potage_names, key=criterion)
    result['chosen_overall_by_validation'] = min(fitted, key=criterion)
    if evaluate_test:
        for name, (model, transform) in fitted.items():
            prediction = model.predict(transform(X[test]))
            result['models'][name]['test'] = score(y[test], prediction)
            print(name, 'test', result['models'][name]['test'], flush=True)
        selected_name = result['chosen_potage_by_validation']
        model, transform = fitted[selected_name]
        whole = model.predict(transform(X[test[:24]]))
        singles = np.array([model.predict(transform(X[[i]]))[0] for i in test[:24]])
        np.testing.assert_allclose(whole, singles, rtol=1e-10, atol=1e-8)
        result['selected_recipe_batch_max_abs_difference'] = float(np.max(np.abs(whole-singles)))
    return result


def save_plot(result, path):
    import matplotlib.pyplot as plt
    if not result['protocol']['test_evaluated']:
        raise ValueError('Plot requires --evaluate-test')
    names = list(result['models'])
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = np.arange(len(names))
    for delta, split, color in [(-0.17, 'validation', '#8ca3b8'), (0.17, 'test', '#247b85')]:
        ax.barh(positions+delta, [result['models'][name][split]['r2'] for name in names],
                height=.32, label=split.capitalize(), color=color)
    ax.set_yticks(positions, [name.replace('potage_', 'Potage: ').replace('_',' ') for name in names])
    ax.invert_yaxis()
    ax.set_xlabel('R² · later months, observed weather')
    ax.set_title('Bike sharing: chronological feature-recipe comparison')
    ax.legend(loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', default='.cache')
    parser.add_argument('--recipes', nargs='+', choices=RECIPES, default=RECIPES)
    parser.add_argument('--select', type=int, default=24)
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.select < 1:
        parser.error('--select must be positive')
    if args.plot and not args.evaluate_test:
        parser.error('--plot requires --evaluate-test')
    result = run_benchmark(args.data_home, args.recipes, args.select, args.evaluate_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.plot:
        save_plot(result, args.plot)


if __name__ == '__main__':
    main()
