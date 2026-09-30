> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Scenario-effect stability probe

This repeats the paired speed and starting-position effect models at seeds 19,
41, and 97. Each run uses 600 paired games, nine snapshot phases, grouped
game splits, and the same `carrier_am`/correlation recipe. The table reports
mean ± standard deviation over seeds on held-out game groups.

| Intervention | Model space | Brier | ROC-AUC | Scenario terms selected |
|---|---|---:|---:|---:|
| Speed +1 | Coarse context | 0.1178 ± 0.0073 | 0.7806 ± 0.0804 | 0.0 |
| Speed +1 | Core state | 0.1198 ± 0.0022 | 0.7749 ± 0.0381 | 0.0 |
| Speed +1 | Core + scenario | 0.1210 ± 0.0050 | 0.7703 ± 0.0520 | 7.0 |
| One-step center start | Coarse context | **0.0313 ± 0.0054** | **0.8324 ± 0.0745** | 0.0 |
| One-step center start | Core state | 0.0317 ± 0.0055 | 0.7983 ± 0.0974 | 0.0 |
| One-step center start | Core + scenario | 0.0324 ± 0.0057 | 0.7823 ± 0.1200 | 11.0 |

Exact selected-expression consensus was effectively absent: no top feature
appeared in all three seeds. Only one coarse-context expression recurred twice
for the position effect; the explicit scenario expressions were different in
each run despite being selected frequently.

## Interpretation

The causal effects are repeatable, but the discovered formulas are not. This
is a useful boundary for Potage in this setting: it can rank benefit above a
coarse baseline, but selected expressions should not yet be interpreted as
stable mechanistic win factors. The explicit speed/start vocabulary is the
least reliable space, with both worse average calibration and higher feature
churn.

Reproduce with:

```bash
python examples/realtime_scenario_effect_stability.py \
    --pairs 600 \
    --seeds 19 41 97 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_scenario_effect_stability_results.json \
    --plot examples/realtime_scenario_effect_stability_results.png
```
