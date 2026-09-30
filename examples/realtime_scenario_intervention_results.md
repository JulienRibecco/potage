> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Causal speed and initial-position interventions

This probe compares matched games with the same skills, latency draws, policy
noise, and opponent parameters. The treated continuation changes only one
factor for the focal player:

- `speed`: movement speed 1 → 2;
- `initial_position`: far corner → a start one cell from the high-value center
  node.

## Results

Each row contains 1,000 matched pairs; all confidence intervals are bootstrap
95% intervals for the paired win-rate effect.

| Intervention | Seed | Focal | Control win rate | Treated win rate | Effect | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| speed | 19 | +1 | 0.670 | 0.745 | +0.075 | [+0.049, +0.102] |
| speed | 19 | -1 | 0.329 | 0.474 | +0.145 | [+0.115, +0.174] |
| speed | 41 | +1 | 0.700 | 0.759 | +0.059 | [+0.033, +0.087] |
| speed | 41 | -1 | 0.329 | 0.464 | +0.135 | [+0.108, +0.161] |
| speed | 97 | +1 | 0.648 | 0.745 | +0.097 | [+0.071, +0.123] |
| speed | 97 | -1 | 0.328 | 0.455 | +0.127 | [+0.101, +0.154] |
| initial position | 19 | +1 | 0.671 | 0.734 | +0.063 | [+0.043, +0.085] |
| initial position | 19 | -1 | 0.367 | 0.526 | +0.158 | [+0.129, +0.187] |
| initial position | 41 | +1 | 0.697 | 0.747 | +0.050 | [+0.028, +0.072] |
| initial position | 41 | -1 | 0.373 | 0.526 | +0.153 | [+0.127, +0.182] |
| initial position | 97 | +1 | 0.649 | 0.701 | +0.051 | [+0.031, +0.072] |
| initial position | 97 | -1 | 0.377 | 0.522 | +0.145 | [+0.117, +0.174] |

Mean paired effects are approximately **+0.106 for speed** and **+0.103 for
initial position**. Both agent characteristics therefore have causal leverage,
but their effects are smaller than the previously measured latency effect
(about +0.127). This agrees with the Potage ablation: speed and initial geometry
are useful modifiers, not dominant standalone factors.

Reproduce with:

```bash
python examples/realtime_scenario_intervention.py \
    --pairs 1000 \
    --seeds 19 41 97 \
    --output examples/realtime_scenario_intervention_results.json \
    --plot examples/realtime_scenario_intervention_results.png
```
