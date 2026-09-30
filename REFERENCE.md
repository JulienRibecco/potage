# Potage API and behavior reference

Potage composes candidate features, selects subsets, and records how to replay
those features on new rows. Start with the [README](README.md) and
[offline example](examples/quick_start.py); this reference covers the current
research alpha. `Pipeline` and `Config` are the primary names. `SoupPipe` and
`SoupConfig` are compatibility aliases.

## Pipeline construction

```python
from potage import Pipeline, Config

pipe = Pipeline(
    X_train, y_train, feature_names,
    feature_types=None,
    config=Config(),
    max_candidates=100_000,
    max_memory='auto',
    X_val=X_val, y_val=y_val,
    fold_eval_count=3,
    fold_seed=42,
    task='regression',
    y_base=None, y_val_base=None,
    exclude_families=None,
)
```

Inputs are numeric arrays: `X_train` has shape `(n_samples, n_features)`,
`y_train` has one value per row, and names must be unique and match the columns.
Encode strings and handle missing values before calling Potage. Optional
`feature_types` contains one of `'numeric'`, `'bool'`, or `'categorical'` for
each column; otherwise types are inferred. Categorical values must therefore
already have numeric encodings. Validation and replay use the same column order.

Explicit `X_val` and `y_val` enable validation scoring. `X_val` alone propagates
validation features but does not enable validation scoring. Without explicit
validation labels, the default is three selection folds; set
`fold_eval_count=None` for training-only selection. These internal folds split
rows and reuse generated features and earlier selections. They are adaptive
selection diagnostics, not nested cross-validation of the discovery process.

Always reserve an outer test split. For grouped or temporal data, provide
appropriate explicit training/validation splits and evaluate the complete
workflow on independently held-out groups or periods. Feature construction,
selection, family pruning, and parameter tuning all belong inside that boundary.

## Composing stages

Generation methods return a `FeatureSet` and accept `select` and
`method='omp'`. Use a non-negative selection count, or `select=None` to keep all
surviving candidates after filtering/deduplication. Regression selection may
return fewer features than requested; classification selects per class, so its
final union can exceed the requested count.

```python
raw = pipe.raw(select=20)
am = pipe.am(raw, select=15)
carriers = pipe.carriers(raw, select=15)
pool = pipe.fuse(raw, am, carriers)
selected = pipe.select(pool, select=20, method='greedy')
X_test_features = pipe.transform(X_test)
```

| Method | Behavior |
|---|---|
| `raw(select, method='omp')` | Identity and unary transforms: powers, log, sqrt, tanh, reciprocal, empirical rank, one-hot |
| `am(*sets, select, method='omp')` | Products, differences, ratios, min/max, harmonic mean, normalized difference, log ratio, damping, ReLU difference, powers |
| `carriers(s, select, method='omp', feature_space='linear')` | Sin/cos, triangle, thresholds, band indicators, Gaussian windows |
| `fm(carrier, mod, select, method='omp')` | Frequency- and phase-modulated sine features |
| `fm_nonsmooth(carrier, mod, select, method='omp')` | Modulated triangle and localized pulse features |
| `wm(carrier, mod, select, method='omp')` | Gaussian windows with shifted centers or scaled widths |
| `fields(s, select, method='omp')` | Localized spatial features on numeric column pairs |
| `fuse(*sets)` | Concatenate feature sets without selection |
| `select(fs, select, method='omp')` | Select again from an existing pool |

One input to `am()` generates within-set pairs; multiple inputs generate
cross-set pairs. `raw()` includes unary transforms; it does not mean identity
columns alone. Boolean and categorical inputs have restricted transform
families. Use `exclude_families` to exclude named families from the candidate pools.

Carrier coordinate spaces include `linear`, `arcsinh`, `log`, `log2`, `log10`,
`sqrt`, `cbrt`, `exp`, `logit`, `tanh`, and `pow:<exponent>`. Signed transforms
are used where applicable; see the [carrier implementation](potage/_carriers.py)
for exact definitions. The named expressions identify generated features;
they are not evidence of a causal or physical mechanism.

## Training references and replay

Transforms can depend on fitted statistics: ranges, percentiles, category
levels, empirical ranks, and standard deviations. These training references
are reused for validation and `transform()`. FFT-guided generation also uses
the target; feature generation is not universally parameter-free or unsupervised.

- Rank features use the training empirical CDF, including the same tie
  convention at fit and replay.
- AM ratio and log-ratio stabilizers use training standard deviations.
- AM power features use signed magnitudes. For an input pair `(a, b)`, the
  forward expression is `sign(a) * (abs(a) + 1e-10) ** (s * b_norm)`, where
  `b_norm` uses training minima/maxima and is clipped to `[0, 1]`.
  `s` comes from `Config.power_scales`. The reverse direction is also generated.
- Other transforms may extrapolate beyond their training ranges. Clipping the
  power exponent input does not bound every generated feature.

`pipe.transform(X_new)` returns the matrix for the **last recorded stage**.
It does not fit a predictor or choose the best historical stage. Fit a downstream
estimator on the matching `FeatureSet.X`, then call its `predict()` on that
replayed matrix. Save both objects to reuse predictions.

A `FeatureSet` exposes `X`, `names`, `feature_types`, and `source_names`.
`source_names` is a tuple of frozensets giving the exact raw-column ancestors
of each output column. Attribute assignment is blocked, but the contained
arrays and lists are not deeply immutable; treat them as read-only to preserve
replay consistency. Use handles from the current history of the same pipeline.

Pickle round-trips are supported in a compatible Python/package environment,
including continuation from restored handles. This is not a stable artifact
format across arbitrary future versions. Load only trusted pickle files.
Pipelines retain training matrices and may produce large serialized objects.

## Selection and scores

| Method | Ranking | Reported score |
|---|---|---|
| `omp` (default) | Correlation with training residuals | R² of an intercept-bearing fit, on validation data when available |
| `greedy` | Incremental ridge score, using validation when available | R² on the scoring split |
| `correlation` | Target correlation minus a redundancy penalty | mRMR-style score, not R² |

Internal folds aggregate these criteria across random row splits. At 3,000 or
more surviving candidates, the current fold selector switches OMP/greedy to
correlation selection with a warning to reduce Gram-matrix allocations.
Always inspect the resulting history's score kind, even when requesting OMP.
This fallback cannot supply the R² required by automatic stopping workflows.

```python
stage = pipe.stages[-1]
history = stage.history
if history is not None:
    print(history.score_kind, history.score_split)
    print(history.selected_names())
```

`SelectionHistory.steps` retains the legacy keys `marginal_r2` and
`cumulative_r2`, and the methods with those names remain available. Interpret
them using `score_kind` and `score_split`. OMP's `selection_score` records its
ranking criterion separately from reported R² gains. Correlation histories
must not be read as explained variance.

For classification, manual stages use one-vs-rest selection and union the
selected features. Union histories are not model R² or accuracy. Supply
explicit validation labels or set `fold_eval_count=None`; internal fold
selection currently supports regression only. Use a separate classifier and
outer evaluation to measure predictive performance.

Low-level selection functions accept a `PrimitiveLibrary` and target array:
`residual_select()` (OMP), `greedy_forward_select()`, `correlation_select()`,
and `per_class_select()`. Consult their signatures/docstrings for options such
as regularization, diversity penalties, and lookahead. `PrimitiveLibrary`
supports constant filtering, correlation deduplication, family/complexity
filtering, and correlation summaries.

## Configuration

`Pipeline` copies the supplied `Config`. These are its constructor defaults:

| Field | Default |
|---|---|
| `carrier_freqs` | `[0.5, 1.0, 2.0, 3.0]` |
| `carrier_phase_scale` | `numpy.pi` |
| `calibrate_n_freqs` | `4` |
| `fm_depths` | `[-2.0, -1.0, 1.0, 2.0]` |
| `pm_depths` | `[0.5, 1.0, 2.0]` |
| `wm_center_pcts` | `[30, 50, 70]` |
| `wm_shift_scale` | `1.0` |
| `wm_width_mods` | `[0.5, 1.0, 2.0]` |
| `step_pcts` | `[25, 50, 75]` |
| `pulse_bands` | `[(0.0, 0.33), (0.33, 0.67), (0.67, 1.0)]` |
| `gauss_center_pcts` | `[25, 50, 75]` |
| `gauss_widths` | `[0.2, 0.5]` |
| `top_k_modulation` | `15` |
| `max_ratios_per_feature` | `None` |
| `power_scales` | `[0.5, 1.0, 2.0, 3.0]` |
| `field_pct_levels` | `[20, 40, 60, 80]` |
| `field_scales` | `[0.1, 0.3, 1.0]` |

Sine/cosine phase is `k * carrier_phase_scale * x_norm`. With the default π
scale, `k=2` means one complete cycle over the training range. Use `2 * numpy.pi`
for frequency values expressed in cycles. `carrier_freqs='fft'` calibrates a
shared grid from training features and targets during pipeline construction.
`fft_carriers(s, select, ...)` instead detects frequencies per input feature.
Neither procedure should see outer test targets.

## Workflows and branching

`auto_run(select=20, max_stage=5, method='omp', early_stop_marginal=0.003,
patience=2, verbose=False)` progresses through raw → AM → carriers → FM/PM →
nonsmooth FM → WM, combining and selecting intermediate pools. `max_stage=0`
stops at raw features. It returns the last accepted stage, which may not be the
best validation checkpoint.

`deepen(base_fs, select=25, max_depth=2, n_carry=15, ...)` recursively composes
FM/PM/WM on carried features with smaller grids and gain-based stopping.
Both workflows require regression with OMP or greedy and R² histories.
Deeper composition is an experiment choice, not a promised improvement.

`prune_run(select=20, max_stage=5, method='omp', n_seeds=3, subsample=0.5)`
runs automatic discovery on training subsamples. It returns `keep`, `exclude`,
and `family_counts`; a family survives if selected in at least one seed.
It requires enough training rows for its minimum subsample size of 50 and
inherits automatic discovery's regression/R² restrictions. Reporting only
family-level results does not prevent leakage or replace outer evaluation.

`discover_carrier_space(fs_raw, select=15, method='omp', spaces=None,
include_raw=True)` compares coordinate spaces and returns a selected
`FeatureSet` plus a report with `winner_counts` and `spaces`. Defaults sweep
linear, arcsinh, log, sqrt, cbrt, and tanh spaces.

For residual discovery, `y_base` and `y_val_base` supply baseline predictions.
Selection targets the corresponding residuals; fit a residual readout and add
its predictions to the baseline at inference. Passing baseline predictions
does not make the workflow cross-fitted. Training, validation, and test
predictions must respect the experiment's split boundaries.

Checkpoints let you compare manual branches:

```python
checkpoint = pipe.checkpoint()
trial = pipe.fm(carriers, raw, select=15)
print(pipe.recap())
pipe.rewind(checkpoint)
alternative = pipe.wm(carriers, raw, select=15)
```

Treat checkpoint tokens as opaque. `rewind()` without an argument undoes one
stage. Rewind restores stage history, budget bookkeeping, family census, and
surviving handles; handles belonging only to discarded stages cannot be reused.
Failed individual stages also restore this bookkeeping. A workflow containing
several completed stages is not a single all-or-nothing transaction.

For repeated discovery excluding consumed raw inputs, see `multi_pass()` and
recipes such as `recipe_am()` in the [preset module](potage/_presets.py).
Exclusion uses exact `source_names` ancestry rather than parsing feature labels.

## Budgets and inspection

`max_candidates=100_000` caps cumulative estimated candidates across generation
steps. `max_memory='auto'` uses half of detected available memory; a float in
`(0, 1]` specifies a fraction, an integer specifies bytes, and `None` disables
the memory guard. Memory detection is best-effort and has platform fallbacks.

`CandidateBudgetExceeded` and `MemoryBudgetExceeded` report estimated budget
violations. These guards are not operating-system memory limits or guarantees
against allocation failures. Filtering, deduplication, selection, retained
training matrices, and optional helpers also affect resource use.

| Inspection | Result |
|---|---|
| `predict_candidates(step, *inputs, select=None)` | Candidate breakdown and selection-pressure estimates |
| `estimate_memory(step, *inputs)` | Estimated peak and component allocations |
| `predict_pipeline_memory(max_stage=5, select=20)` | Estimated automatic workflow and budget fit |
| `stages` | Shallow copy of completed `Stage` records |
| `recap()` | Stage summary with candidates, selected counts, scores, methods, timings |
| `unused_families()` | Generated families never selected in recorded history |
| `stage.detail()` | Candidate funnel, selection scores, and family counts |

`Stage.final_r2` and `Stage.diagnosis` are available only for nonempty R²
histories; otherwise they are `None`. `Stage.curve` exposes legacy score values,
so check the history metadata. `family_census` counts families in the selection
history; stages without selection history have an empty census.

## Package organization and optional integrations

| Modules | Responsibility |
|---|---|
| `_pipe`, `_records` | Public pipeline orchestration, feature handles, stage records |
| `_replay`, `_state` | Replay, serialization lookup, checkpoints, rollback |
| `_dispatch`, `_select`, `_kfold` | Selection routing, algorithms, fold diagnostics |
| `_config`, `_algebra`, `_carriers`, `_frequency`, `_modulation`, `_fields` | Transform configuration and builders |
| `_feature_utils`, `_provenance` | Shared transform helpers and raw-column ancestry |
| `_budget`, `_limits` | Candidate/memory estimates and shared thresholds |
| `_library`, `_dedup`, `_diagnose` | Candidate containers, deduplication, diagnostics |
| `_workflows`, `_presets` | Automatic strategies and reusable recipes |
| `_stages` | Standalone `apply_stage()` and compatibility exports |

Use exports from `potage` for application code; underscore modules are internal.
`probe_fields()` and `neuron_probe_groups()` require the separate legacy
`signalfault` neural helper. Optional native builders in `potage._accel`
require an externally built `libsoup` binary and do not accept Python `Config`.
The legacy `c_omp_select()` wrapper uses Python OMP without that binary.

The [changelog](CHANGELOG.md) records validation and limitations. The
[research catalog](docs/research-catalog.md) distinguishes historical reports
from rerun experiments; the [AM replay audit](research/am-replay-audit/README.md)
records the focused investigation of the earlier reference-statistics defect.
