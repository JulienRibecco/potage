# Potage — composable feature discovery for tabular research

Potage builds candidate features from tabular data, selects useful combinations,
and replays the chosen transformations on new rows. It grew out of scientific
experiments with products, periodic signals, and localized effects.

The engineering focus is a reusable experiment component: compose stages,
inspect named features, control candidate and memory budgets, and reuse the
result with ordinary sklearn estimators. Research alpha; APIs may evolve.

## Install and try it

Python 3.9 or newer, from this repository:

```bash
pip install -e .
python examples/quick_start.py
```

The demo needs no download or optional dependencies. It builds an explicitly
synthetic nonlinear target, discovers features on training/validation data, and
compares raw ridge with Potage + ridge on a separate test split. This shows the
workflow, not a general performance advantage.

## Compose a feature pipeline

```python
from potage import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Split your data first. Validation is used for feature selection.
pipe = Pipeline(X_train, y_train, feature_names,
                X_val=X_val, y_val=y_val)
raw = pipe.raw(select=10, method='greedy')
carriers = pipe.carriers(raw, select=10, method='greedy')
merged = pipe.fuse(raw, carriers)
selected = pipe.select(merged, select=12, method='greedy')

model = make_pipeline(StandardScaler(), Ridge(alpha=1))
model.fit(selected.X, y_train)
predictions = model.predict(pipe.transform(X_test))
print(selected.names)
```

The returned `FeatureSet` holds a feature matrix and names; it can feed another
stage or an estimator. `selected.source_names` gives the exact raw-column
ancestors of each feature; multi-pass recipes use this to exclude consumed
inputs without guessing from their labels. `transform()` returns the feature
matrix for the final recorded stage. Save both the pipeline and downstream
estimator to reuse predictions. Python pickle round-trips are supported in a
compatible environment; load only trusted files.

| Stage | Candidate features |
|---|---|
| `raw()` | Identity, powers, log, reciprocal, empirical rank, one-hot |
| `am(fs)` | Products, ratios, differences, powers |
| `carriers(fs)` | Sin/cos, triangle, steps, pulses, Gaussian windows |
| `fm(carrier, mod)` | Frequency and phase modulation |
| `fm_nonsmooth(carrier, mod)` | Triangle and localized pulse modulation |
| `wm(carrier, mod)` | Moving or widening Gaussian windows |
| `fuse()` / `select()` | Combine pools / choose a subset |

See the [API and behavior reference](REFERENCE.md) for configuration, replay,
selection scores, and advanced workflows. `SoupPipe` and `SoupConfig` remain
compatibility aliases for `Pipeline` and `Config`.

## Evaluation and practical limits

- Keep a test set outside feature discovery. Repeated selection against a
  validation set can overfit it. Named expressions are associations, not proof
  of physical mechanisms or causality.
- Without explicit validation, `fold_eval_count=3` provides selection
  diagnostics. Greedy ranks by validation R²; OMP ranks by training-residual
  correlation and reports validation R². Correlation selection reports an
  mRMR-style score, **not R²**. Check `history.score_kind` and
  `history.score_split`; step field names retain their legacy R² spelling.
  These folds reuse generated features and prior selections; they are not
  nested CV.
- Internal folds split rows. For grouped or temporal problems, provide a
  suitable explicit validation split and independently evaluate the entire
  feature-discovery pipeline on outer folds. `task='classification'` currently
  requires `fold_eval_count=None` unless explicit validation labels are supplied.
- Learned reference statistics are reused when replaying transforms. Rank ties
  use the same empirical-CDF convention at fit and transform. AM power
  transforms normalize exponent inputs using training minima/maxima, clip
  them to [0, 1], then apply the configured exponent scales. Other unbounded
  transforms can still extrapolate badly. Deeper composition is not
  automatically better.
- `auto_run()` returns its last accepted stage, not necessarily the best
  validation checkpoint. Use explicit stages/checkpoints for model selection.
  `auto_run()` and `deepen()` use R² stopping and require regression with OMP
  or greedy selection. Failed individual stages restore their bookkeeping;
  `rewind()` also restores budget estimates and surviving feature handles.
- Candidate and memory guards are estimates, not hard process memory limits.
  The pipeline retains training matrices for replay and can be large.

## Research examples

Start with the [concrete-strength benchmark](examples/concrete_strength_results.md):
recipe-grouped splits, conventional baselines, and a progression of feature
stages. Install the optional example dependencies to download data or plot:

```bash
pip install -e '.[examples]'
python examples/concrete_strength.py --max-stage 1
```

The [research catalog](docs/research-catalog.md) preserves the broader game,
intervention, and transport investigations. Those saved results predate the
replay/scoring fixes unless explicitly marked as rerun. They are exploratory
records, not current release benchmarks.

## Validation

Run the regression suite with `python -m unittest discover -s tests -v`.
The [changelog](CHANGELOG.md) records fixes and validation; the
[AM replay audit](research/am-replay-audit/README.md) documents a past defect
and its effect on research results.

## Dependencies and license

Core: NumPy, SciPy, scikit-learn. No deep learning framework.
`neuron_probe_groups()` / `probe_fields()` additionally require the legacy
`signalfault` neural helper; they are not part of the standalone quick start.
Optional native builders in `potage._accel` require an externally built
`libsoup` binary and use native defaults; they do not accept Python Config.
The legacy `c_omp_select()` name wraps Python OMP and needs no native binary.

[CC BY 4.0](LICENSE).
