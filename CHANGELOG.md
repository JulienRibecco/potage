# Potage changelog

## 0.1.1 — 2026-09-30 (research alpha)

- Added a reproducible chronological bike-sharing comparison covering eight
  current Potage recipes, raw/calendar/cyclical ridge, and histogram boosting.
  Feature/penalty/recipe choices are made on validation before test reporting.
- Fixed field candidate filtering to use training-reference variability during
  validation and replay, preventing mismatched columns and batch-dependent
  indicators. Added two field regressions and three benchmark protocol tests.
- Local Python 3.10 suite: 68 tests passed; the installed wheel passed 37
  contract tests outside the checkout. All eight bike recipes pass fitted
  matrix replay and validation batch checks; saved scores retain every baseline
  and recipe, including the validation winner's poorer test result.

- Reworked the research catalog around validation status and experiment entry
  points, removing obsolete installation/API material.
- Added clone and virtual-environment setup to the README and clarified the
  experimental purpose of FFT-guided carrier discovery.
- Made concrete the standalone AM audit example and documented the optional
  metallic-glass table's local provenance, schema, hashes, and availability
  limits. Recorded numerical results and their original source hashes are
  preserved; their hashes describe the code used in those earlier runs.

## 0.1.0 — 2026-09-30 (research alpha)

- AM ratios/log ratios and power transforms reuse training statistics in both
  single-set and cross-set validation/replay. Power transforms normalize
  exponent inputs using training minima/maxima, clip them to [0, 1], then
  apply the configured scales. Batch size and ordering no longer change a
  row's features in these paths.
- Rank features use a consistent empirical CDF, including ties. Noninteger
  categorical values receive distinct one-hot names.
- OMP retains training-residual feature ranking but reports actual validation
  R² when a validation set or fold is available. Fits include an intercept.
  `select()` now honors fold scoring as the generation methods do. Invalid
  fold counts fail clearly. Selection folds are not independent outer CV.
- Pickled pipelines reload successfully and reconstruct feature-handle lookup
  tables so work can continue after loading.
- Added regression coverage for replay, validation propagation, serialization,
  batch/permutation invariance, and agreement with independent sklearn fits.
- Added a no-download quick start, standalone wheel metadata, license text,
  optional example dependencies, and CI for core and installed-package tests.
- Replaced removed NumPy `row_stack` aliases with `vstack` in research
  examples so dataset generation works with current NumPy releases.
- Corrected the concrete example for current sklearn APIs and reran all stages
  with the original split and settings. Refreshed its JSON, figure, and report.

### Correctness and shared behavior

- Generation and explicit `select()` share one selection dispatcher, including
  classification's one-vs-rest union and validation/fold routing.
- Histories expose `score_kind` and `score_split`. Correlation and class-union
  scores are no longer reported or diagnosed as R². OMP marginal values now
  represent R² gains; ranking correlations live in `selection_score`.
  R²-based `auto_run()` and `deepen()` require regression with OMP or greedy.
- Standalone and cross-set AM share the same builder and training references.
  Cross-set candidate ordering now follows that common builder; ties can select
  a different representative than before.
- Failed stages restore candidate estimates, memory estimates, and family
  census. Rewind restores surviving handles, including single-input fusion.
  Checkpoint tokens are opaque: pass them to `rewind()` rather than unpacking.
- Features carry exact raw-column ancestry exposed through
  `FeatureSet.source_names`; multi-pass exclusion no longer parses labels.
  Names remain string-compatible and ancestry survives pickle.
- Fixed multi-pass budget overrides and empty-pass errors; made dedup
  representative policy consistent across clustering methods. Invalid method,
  selection, fold, and feature-type arguments fail clearly.
- Fixed deepening replay when the carry set is a subset of its parent.
- Split pipeline records, replay, state, selection routing, and budget estimates
  into focused internal modules. Removed unused normalization/cache helpers,
  imports, intermediate arrays, and duplicate union-find regrouping.
- Retained `SoupPipe` / `SoupConfig` aliases and optional native builder entry
  points. `c_omp_select()` correctly works without a native binary; native
  builders reject unsupported Python Config instead of silently ignoring it.

### Internal module organization

- Split stage implementations into `_config`, `_algebra`, `_carriers`,
  `_frequency`, `_modulation`, `_fields`, and `_feature_utils`. `_stages` keeps
  standalone `apply_stage()` and compatibility exports. Candidate counting
  now lives alongside the other budget estimates.
- Extracted automatic progression, family pruning, recursive deepening, and
  carrier-space discovery into ordinary functions in `_workflows`. Existing
  Pipeline methods retain their signatures and documentation and delegate to
  these functions; no additional mixins were introduced.
- Preserved all 31 extracted stage function/class bodies. Compared all four
  workflows before and after the split: selected features, matrices, replay,
  histories, ancestry, budgets, and metadata matched exactly. A pipeline
  pickled before the split loaded and replayed successfully afterward.

### Recorded local validation (2026-09-30)

Python 3.10: 63 tests passed, including 21 refactoring regressions and five
workflow/import contracts. The built wheel passed all 35 installed-package
contract tests from outside the checkout and ran the offline quick start. The full concrete rerun reproduced every saved
validation/test metric within 1e-10; its source hashes were refreshed.
The external native library and signalfault probe integration were not run.
CI is configured to repeat the package checks on Python 3.9 and 3.13.

### AM replay audit

The [AM-only research audit](research/am-replay-audit/README.md) records paired
experiments, batch-invariance checks, and unaffected-family controls for the
reference-statistics fix. Its script, results, and figure live under `research/`;
usage examples remain under `examples/`.

### Existing research records

The concrete benchmark was rerun with the current library. The focused AM audit
also compares concrete and metallic-glass experiments; it does not revalidate
the historical full metallic-glass cascade. The game/intervention/transport
JSON and PNG files remain historical evidence, not current benchmark claims.
Their Markdown reports and the research catalog now state this status.
Rerun them with the corrected library before interpreting feature rankings,
stability, or causal/transport conclusions. Simulator unit tests passing does
not validate the saved statistical results.

### Documentation

- Replaced the outdated `SOUP_V2.md` guide with the
  [Potage API and behavior reference](REFERENCE.md). Updated defaults, scoring,
  replay, workflow limits, and the module map against the current code.
- The README uses the primary `Pipeline` name and links the reference; legacy
  `SoupPipe` / `SoupConfig` aliases remain supported.

### Limits retained for this alpha

Explicitly split grouped/temporal data before discovery and use outer evaluation.
Correlation-selection history fields retain legacy R² names but contain mRMR
scores. Automatic stage progression does not restore the best validation stage.
Some optional probe helpers still depend on signalfault, and native acceleration
requires a separately built library. Model quality, causality, and physical
interpretability remain questions for the experiment, not library guarantees.
