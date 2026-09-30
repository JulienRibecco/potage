> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Synthetic game-state win-factor probe

## Question

Can Potage identify conditional win factors from raw game-state variables, and
do those factors improve win probabilities on untouched matches?

This probe uses 6,000 independent match snapshots. The visible table contains
23 state variables, including six pure noise columns. Outcomes are sampled from
a hidden logistic law containing four planted factors:

1. Relative gold: `log(our_gold / opponent_gold)`
2. Objective pressure combined with the opposing side's health
3. Counter units matched against the opposing side's heavy units
4. Map-control advantage modulated by game time

Potage never sees those derived factors. An obvious-state logistic baseline
receives time, score difference, and alive-player difference. Potage searches
for features that explain the baseline's cross-fitted residual. Unsafe
reciprocal, direct ratio, and power families are excluded, and the search stops
after the AM stage.

The fixed split is 3,600 discovery, 1,200 validation, and 1,200 untouched test
matches.

## Primary result

| Model | Test Brier ↓ | Test log loss ↓ | Test ROC-AUC ↑ |
|---|---:|---:|---:|
| Obvious-state baseline | 0.2372 | 0.6665 | 0.6222 |
| Baseline + selected raw state | 0.2130 | 0.6141 | 0.7208 |
| Baseline + Potage AM | **0.1659** | **0.5011** | **0.8348** |
| Oracle with planted factors | 0.1584 | 0.4821 | 0.8490 |

AM closes about 90% of the Brier-score gap between the obvious baseline and
the oracle. Validation and untouched-test scores are close, so this gain does
not show the divergence observed in the deep concrete experiment.

## Recovered factors

All four planted relationships appear among the 16 selected AM features:

| Planted mechanism | Selected evidence |
|---|---|
| Relative gold | `our_gold/(1+|opponent_gold|)` |
| Counter matchup | `our_counter_units*opponent_heavy_units` and the opposing counterpart |
| Objective finish | health/objective-pressure coupled features in both directions |
| Time-dependent map value | normalized time/map-control interactions for both sides |

Three additional complete simulations reproduced the result:

| Seed | Baseline Brier | Potage AM Brier | Oracle Brier | Factors recovered |
|---:|---:|---:|---:|---:|
| 19 | 0.2411 | 0.1682 | 0.1572 | 4/4 |
| 41 | 0.2353 | 0.1653 | 0.1558 | 4/4 |
| 97 | 0.2363 | 0.1710 | 0.1564 | 4/4 |

## Interpretation

The idea passes this controlled probe: shallow compositional features recover
the variables participating in hidden conditional win factors and yield a
large out-of-sample probability improvement.

It does not recover a unique symbolic law. Several selected gold × map-control
terms reconstruct the time × map effect because total gold rises with time in
the simulator. That is a useful warning for real telemetry: Potage can expose a
predictive proxy while leaving the analyst to distinguish it from the actual
mechanism. One low-weight noise interaction also survives the single primary
split, reinforcing the need for stability selection on real data.

## Library issue found by the probe

The initial run silently lost bounded health and map-control variables when
unary families were excluded. Potage performed correlation deduplication before
family exclusion, sometimes chose an excluded transform as a cluster's
representative, and then removed the whole cluster. Family exclusion now runs
before deduplication, and a regression test preserves this behavior.

## Next real-data probe

Use the same structure with one or more snapshots per actual match:

- Split by match, player, patch, and map where possible.
- Cross-fit the obvious win-probability baseline.
- Search only raw and AM families first.
- Retain factors recurring across grouped folds.
- Report Brier score, calibration, and log loss, not only ROC-AUC.
- Treat selected factors as predictive associations until action-level or
  matched-state analysis supports a causal interpretation.

Exact results and selected features are stored in
`examples/game_state_probe_results.json`.
