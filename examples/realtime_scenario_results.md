> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Initial-state and agent-characteristic probe

This probe varies four initial geometries (`diagonal`, `horizontal`, `vertical`,
and `center_edge`) and four speed pairs (`1-1`, `2-1`, `1-2`, `2-2`). The
simulation emits the original 20 state features plus six explicit scenario
descriptors: both agent speeds, speed difference, both initial distances to the
high-value center node, and initial separation.

## Main run

The 1,500-game run produced 1,487 complete games and 26,766 snapshots. The
baseline test Brier was `0.0910`.

| Recipe | Feature space | Test Brier | Test ROC-AUC | Scenario features selected |
|---|---|---:|---:|---|
| `raw` | core | 0.0905 | 0.9500 | none |
| `raw` | scenario | 0.0911 | 0.9494 | start distances, both speeds |
| `carrier_am` | core | 0.0900 | 0.9509 | none |
| `carrier_am` | scenario | **0.0899** | 0.9511 | initial distance/target interactions |

The direct speed/start columns enter the raw selected set, but they do not
generalize as standalone predictors. With composition, Potage turns them into
small interactions such as initial distance × target distance and speed
difference × score regime, yielding a slight held-out gain.

## Stability check

On three additional 400-game seeds, the scenario-feature `carrier_am` recipe
won on two seeds and lost to the core version on one. The raw scenario recipe
was worse than raw core on all three. This suggests that agent speed and initial
geometry are useful mainly as conditional modifiers, not as unconditional main
effects.

Reproduce the main run with:

```bash
python examples/realtime_scenario_probe.py \
    --games 1500 \
    --recipes raw carrier_am \
    --output examples/realtime_scenario_results.json \
    --plot examples/realtime_scenario_results.png
```

Run the smaller stability check with:

```bash
python examples/realtime_scenario_probe.py \
    --games 400 \
    --recipes raw carrier_am \
    --stability-seeds 19 41 97 \
    --stability-games 400 \
    --output examples/realtime_scenario_stability_results.json
```
