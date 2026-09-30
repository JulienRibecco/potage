> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Scenario-effect transport probe

This probe discovers an effect model on one intervention magnitude and tests it
on a different magnitude from a fresh randomized geometry sample:

- train on focal speed `+1`, test on focal speed `+2`;
- train on a one-step move toward the center, test on a two-step move.

All snapshots from each paired game remain in one group. The intervention
target is a loss-to-win switch in the treated replay.

## Result

Run: 800 training pairs per intervention, seed 1401, nine phases,
`carrier_am` with correlation selection.

| Held-out intervention | Model space | Brier | ROC-AUC |
|---|---|---:|---:|
| Speed `+2` | Coarse context | **0.1446** | 0.7933 |
| Speed `+2` | Core state | 0.1457 | **0.7947** |
| Speed `+2` | Core + scenario | 0.1533 | 0.7627 |
| Center move `2` | Coarse context | **0.0351** | 0.7579 |
| Center move `2` | Core state | 0.0365 | **0.7749** |
| Center move `2` | Core + scenario | 0.0370 | 0.7635 |

The held-out positive-switch rate is higher than training for both effects
(21.5% versus 14.0% for speed; 4.0% versus 2.5% for position), so Brier scores
also reflect a calibration shift. ROC-AUC provides the cleaner transport
comparison.

## Interpretation

Current-state features retain modest ranking power when the intervention
magnitude changes. Explicit speed/start descriptors do not transport: they
select several scenario expressions, but those expressions lower held-out
ranking and calibration. This strengthens the earlier conclusion that speed and
initial geometry affect outcomes through evolving control/tempo state rather
than as stable standalone win factors.

Reproduce with:

```bash
python examples/realtime_scenario_effect_transport.py \
    --pairs 800 \
    --seed 1401 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_scenario_effect_transport_results.json \
    --plot examples/realtime_scenario_effect_transport_results.png
```
