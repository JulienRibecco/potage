> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Connect Four win-factor probe

Connect Four tests whether the analysis transfers from Reversi's smooth
positional factors to discrete tactical structure. The local generator records
game snapshots from the player-to-move perspective and predicts the eventual
winner from tactical state descriptors.

## Protocol

- 3,000 stochastic self-play games; 2,867 non-draw games retained.
- 9 snapshots per game when the game reaches the corresponding ply.
- 9,877 snapshots total, split by complete game.
- Baseline: phase, disc lead, player-skill difference, and first-player color.
- State descriptors: legal columns, immediate wins, double threats, open threes,
  center/edge control, runs, stack heights, and parity.
- Potage excludes unary and high-flexibility ratio/power families for the main
  AM comparison.

## Main result

| Model | Validation Brier | Test Brier | Test ROC-AUC |
|---|---:|---:|---:|
| Obvious-state baseline | 0.1369 | 0.1291 | 0.9018 |
| Potage raw state | 0.1226 | 0.1161 | 0.9198 |
| Potage shallow AM | **0.1188** | **0.1142** | **0.9216** |
| Full-state histogram tree | 0.1280 | 0.1157 | 0.9200 |

Raw state is the reliable gain: it improves the baseline in every alternate
seed. Shallow AM adds 0.0020 Brier improvement over raw on the primary split,
but the incremental gain is smaller and less stable than in Reversi.

## Tactical factors found

The main run's highest-weight AM terms are combinations of:

- our double threats and open threes;
- opponent double threats and open threes;
- edge discs and center control;
- playable-column count and existing runs.

This is the expected Connect Four signature: a position becomes strong when it
creates multiple future wins or forces the opponent to answer a threat.

## Seed stress test

| Seed | Baseline | Raw | AM | Tree |
|---:|---:|---:|---:|---:|
| 19 | 0.1351 | **0.1319** | 0.1293 | 0.1468 |
| 41 | 0.1339 | **0.1191** | 0.1217 | 0.1590 |
| 97 | 0.1343 | **0.1213** | 0.1203 | 0.1440 |

AM improves raw in two of three runs. Exact formulas move between seeds, but
open-threes and double-threat terms recur. This is a useful warning: tactical
composition is more sensitive to the self-play distribution than Reversi's
ingredient-level signal.

The next Connect Four-specific step would be an intervention on double-threat
creation versus open-three creation, with paired games and role swaps. Until
that is run, these are predictive tactical correlates—not established causal
factors.

## Reproduce

```bash
python examples/connect_four_probe.py \
    --output examples/connect_four_probe_results.json \
    --plot examples/connect_four_probe_results.png \
    --stability-seeds 19 41 97
```
