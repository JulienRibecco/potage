> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Unseen-scenario cross-fit

The pooled scenario probe showed that speed and initial distance can enter
selected expressions. This experiment asks the stricter question: do those
expressions transport to an initial geometry that was completely held out?

Each of the four scenario geometries is held out once. The remaining three
scenarios are split into discovery and validation games; the held-out geometry
is never used for feature selection.

## Results

The 800-game run used `carrier_am` + correlation selection:

| Feature space | Mean held-out Brier | Mean held-out ROC-AUC |
|---|---:|---:|
| Core 20 state features | **0.0874** | **0.9492** |
| Core + speed/start descriptors | 0.0884 | 0.9483 |
| Core + invariant time-to-center descriptors | 0.0884 | 0.9480 |

Scenario descriptors were selected in the held-out-fold pipelines—especially
initial distances and speed difference—but both the raw and normalized expanded
spaces were slightly worse on unseen geometries. This is a useful negative
result: selected terms can describe a scenario without being invariant across
scenarios, and simple time-to-center normalization does not solve transport by
itself.

The implication is that Potage should see a deliberately diverse scenario
distribution before we treat speed/start interactions as transferable factors.
Within-scenario causal effects remain strong; cross-scenario transport is the
current bottleneck.

A second 800-game seed reproduced the direction: core Brier `0.0857`, raw
scenario `0.0865`, and invariant `0.0867`. Thus the transport gap is not a
single-seed artifact.

Reproduce with:

```bash
python examples/realtime_scenario_crossfit_probe.py \
    --games 800 \
    --recipe carrier_am \
    --output examples/realtime_scenario_crossfit_results.json \
    --plot examples/realtime_scenario_crossfit_results.png
```
