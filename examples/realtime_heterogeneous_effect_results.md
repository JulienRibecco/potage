> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Heterogeneous latency-benefit probe

This experiment asks where the causal latency effect is largest. For each game
and focal player, the same skills, opponent latency, initial state, and policy
noise are replayed at latency 0 and latency 3. The reference features come from
the latency-3 trajectory. The target is whether the low-latency replay flips a
loss into a win. Game IDs stay grouped across all nine snapshot phases.

The main run used 1,000 matched pairs, 17,739 rows after removing draw or
mismatch pairs, and a 20% untouched game split. The robust `carrier_am` +
`correlation` recipe was used.

| Model | Test Brier | Test log loss | Test ROC-AUC |
|---|---:|---:|---:|
| Coarse time/score context | 0.1322 | 0.4171 | 0.7736 |
| Potage effect model | **0.1009** | **0.3272** | **0.8738** |

The strongest expressions are nonlinear thresholds and interactions involving
`score_difference`, controlled nodes, and distance. In this arena, the main
heterogeneity signal is therefore not simply “more pending time is bad”; it is
that latency reduction is most valuable in particular score/control states,
especially near outcome thresholds. The model's mean predicted benefit declines
from 0.228 at tick 10 to 0.151 at tick 90, though the marginal observed benefit
is constant across phases because each pair's eventual win switch is repeated at
every snapshot. Treat the phase curve as conditional model output, not a direct
phase treatment estimate.

Reproduce with:

```bash
python examples/realtime_heterogeneous_effect_probe.py \
    --pairs 1000 \
    --seed 701 \
    --recipe carrier_am \
    --method correlation \
    --output examples/realtime_heterogeneous_effect_results.json \
    --plot examples/realtime_heterogeneous_effect_results.png
```
