> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Simultaneous-action arena probe

This probe asks whether a Potage-style compositional search can recover useful
win factors in a game with real-time controls rather than turns. It uses a small
5x5 arena with three capture nodes. Each agent continuously moves toward a
chosen node, but target commands take a hidden latency of 0--3 ticks to apply.
Nodes generate score while held; the label at each snapshot is which agent wins
the eventual race.

The baseline sees only coarse context: time fraction, current score difference,
skill difference, and player side. The state features expose positions,
distances, capture progress, control counts, score rates, and pending-action
remaining time, but not the hidden latency parameter itself. Rows are grouped
by game for discovery, validation, and an untouched test split.

## Main run

3,000 requested games produced 2,974 non-draw games and 53,532 snapshots. The
test split contains 595 complete games.

| Model | Test Brier | Test log loss | Test ROC-AUC |
|---|---:|---:|---:|
| Coarse baseline | 0.0797 | 0.2488 | 0.9614 |
| Potage raw | 0.0794 | 0.2486 | 0.9616 |
| Potage AM | **0.0789** | **0.2464** | **0.9622** |
| Full-state tree | 0.0717 | 0.2246 | 0.9687 |

The tree is stronger as a predictor, but Potage AM supplies an interpretable
compositional vocabulary. The strongest main-run terms include:

- target distance × pending action time;
- controlled-node count × opponent distance;
- opponent target progress normalized by agent separation;
- score rate combined with control and distance.

## Stability check

Three additional 1,000-game seeds all improved on the coarse baseline with
Potage AM:

| Seed | Baseline | Potage AM | Difference |
|---:|---:|---:|---:|
| 19 | 0.0803 | 0.0760 | -0.0043 |
| 41 | 0.0871 | 0.0853 | -0.0018 |
| 97 | 0.0846 | 0.0832 | -0.0013 |

The exact expressions vary, as expected, but pending-action × distance terms
recur in all three runs. This is evidence for a timing/tempo factor, not proof
that this toy arena captures a general real-time-game law.

Reproduce with:

```bash
python examples/realtime_race_probe.py \
    --output examples/realtime_race_probe_results.json \
    --plot examples/realtime_race_probe_results.png \
    --stability-seeds 19 41 97
```
