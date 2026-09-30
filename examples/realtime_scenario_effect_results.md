> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Heterogeneous scenario-effect probe

This experiment changes the target from final win prediction to a matched
counterfactual: does the focal agent switch from a loss to a win when it is
given one extra movement speed, or when it starts one step closer to the center
node? The two replays share the same random seed and policy noise. Start cells
and base speed pairs are randomized, and all snapshots from a game stay in one
group during discovery and testing.

## Result

Run: 1,000 paired games, seed 1301, nine snapshot phases, `carrier_am` with
correlation selection. Brier and AUC are on untouched game groups.

| Intervention | Model space | Test Brier | Test AUC |
|---|---|---:|---:|
| Speed +1 | Coarse context | 0.1105 | 0.8220 |
| Speed +1 | Core state | **0.1085** | **0.8244** |
| Speed +1 | Core + scenario | 0.1157 | 0.7857 |
| One-step center start | Coarse context | **0.0268** | **0.8620** |
| One-step center start | Core state | 0.0292 | 0.8497 |
| One-step center start | Core + scenario | 0.0303 | 0.8269 |

The positive-switch rates were 13.7% for speed and 2.8% for position. The
scenario model selected many expressions involving speed/start descriptors, but
they reduced held-out performance in both intervention tasks.

## Interpretation

Potage does find a meaningful heterogeneous-effect vocabulary, especially
score difference, control progress, and target pressure. However, the stable
predictive signal is mostly in the current trajectory state, not in explicit
initial speed or geometry columns. The causal intervention can matter while
the intervention metadata remains redundant or selection-unstable.

This suggests the next useful probe is transport: repeat the effect model over
held-out starting geometries and held-out intervention magnitudes, then compare
feature consensus and calibration. A selected scenario term should only be
called a win factor if it predicts paired benefit across those held-out
regimes.

Reproduce with:

```bash
python examples/realtime_scenario_effect_probe.py \
    --pairs 1000 \
    --seed 1301 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_scenario_effect_results.json \
    --plot examples/realtime_scenario_effect_results.png
```
