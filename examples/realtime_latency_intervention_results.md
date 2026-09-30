> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Latency intervention

This is a matched-pairs causal check for the real-time arena. For each pair,
the same skills, opponent latency, initial positions, and pre-generated policy
noise are replayed twice. The focal agent receives either latency 0 or latency
3 for every command; everything else is held fixed. A positive effect means
that latency 0 changes a loss into a win more often than it changes a win into
a loss.

The experiment used 1,000 pairs for each focal player and seed, with seeds
19, 41, and 97. Draw/mismatch pairs were discarded before computing effects.

| Seed | Focal | Latency-0 win rate | Latency-3 win rate | Paired effect | 95% bootstrap CI |
|---:|---:|---:|---:|---:|---:|
| 19 | +1 | 0.782 | 0.650 | +0.133 | [+0.103, +0.163] |
| 19 | -1 | 0.445 | 0.320 | +0.125 | [+0.097, +0.155] |
| 41 | +1 | 0.783 | 0.649 | +0.134 | [+0.107, +0.163] |
| 41 | -1 | 0.420 | 0.315 | +0.106 | [+0.077, +0.136] |
| 97 | +1 | 0.756 | 0.627 | +0.129 | [+0.100, +0.160] |
| 97 | -1 | 0.444 | 0.306 | +0.138 | [+0.110, +0.167] |

The mean paired effect is **+0.127** across the six runs, and every confidence
interval is above zero. Thus command latency is not merely a predictive proxy
in this toy game: reducing latency causes a substantial win-rate advantage under
the matched replay protocol.

Reproduce with:

```bash
python examples/realtime_latency_intervention.py \
    --pairs 1000 \
    --seeds 19 41 97 \
    --output examples/realtime_latency_intervention_results.json \
    --plot examples/realtime_latency_intervention_results.png
```
