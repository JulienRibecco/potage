> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Phase-specific latency branches

This experiment fixes the previous heterogeneity limitation. For each game and
branch phase, both variants share the exact same prefix—including skills,
opponent latency, policy noise, positions, scores, and pending commands. Only
after the branch tick does the focal agent receive latency 0 or latency 3. The
target is whether the low-latency continuation changes a loss into a win.

The run used 500 matched game pairs, nine branch ticks, and the
`carrier_am` + `correlation` Potage recipe. It generated 9,000 requested
branches, kept 8,929 non-draw branches, and observed **zero common-state
mismatches** at the branch point.

## Overall prediction

| Model | Test Brier | Test log loss | Test ROC-AUC |
|---|---:|---:|---:|
| Coarse time/score context | 0.0754 | 0.2685 | 0.7530 |
| Potage effect model | **0.0723** | **0.2452** | **0.8319** |

## Causal effect by branch phase

The observed rate is now a genuine phase-specific treatment effect, because
the intervention begins only after that tick:

| Branch tick | Observed low-latency win switch | Potage predicted benefit |
|---:|---:|---:|
| 10 | **0.186** | 0.169 |
| 20 | 0.145 | 0.125 |
| 30 | 0.136 | 0.110 |
| 40 | 0.080 | 0.053 |
| 50 | 0.105 | 0.041 |
| 60 | 0.085 | 0.036 |
| 70 | 0.040 | 0.043 |
| 80 | 0.025 | 0.029 |
| 90 | 0.005 | 0.022 |

This confirms that command latency has its largest causal leverage early in the
race. The strongest discovered state terms combine target distance, agent
separation, score difference, and capture progress. The mechanism is therefore
“latency matters while there is still time and positional slack to convert,”
not a uniform penalty at all phases.

Reproduce with:

```bash
python examples/realtime_phase_branch_probe.py \
    --pairs 500 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_phase_branch_results.json \
    --plot examples/realtime_phase_branch_results.png
```
