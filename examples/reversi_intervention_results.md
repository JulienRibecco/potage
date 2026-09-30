> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Reversi intervention results

The predictive probes repeatedly identified mobility, frontier exposure,
corners, corner-adjacent C-squares, and available flips. This experiment turns
that vocabulary into an intervention: an otherwise identical stochastic agent
adds either mobility, safety, or their combination to its move score.

## Design

- Each paired observation starts both games from the same random 10-ply board.
- The challenger controls Black in one game and White in the other, removing
  opening-position and color advantage.
- A zero-strength challenger is exactly neutral by construction (paired score
  0.5), which is covered by a regression test.
- Strength is selected using 100 tuning pairs from the grid
  `0, 2, 4, 8, 12, 16, 24, 32`.
- The selected strength is evaluated once on 500 new pairs (1,000 games).
- Confidence intervals bootstrap whole game pairs 2,000 times.

The mobility component favors moves leaving more options for us than the
opponent. Safety favors lower own frontier exposure, corner control, and
C-squares only when their adjacent corner makes them safe.

## Untouched intervention test

| Intervention | Selected strength | Paired score | Gain over 0.5 (95% CI) | Decisions changed |
|---|---:|---:|---:|---:|
| Mobility | 24 | 0.6205 | +0.1205 [+0.0910, +0.1480] | 44.6% |
| Safety | 12 | 0.5795 | +0.0795 [+0.0500, +0.1070] | 16.4% |
| Combined | 24 | **0.6880** | **+0.1880 [+0.1610, +0.2165]** | 51.5% |

Both independently discovered concepts cause higher win scores against the
matched baseline policy, and the combined effect is larger than either
ablation. The mechanism check also passes: the intervention changes decisions
and increases the targeted factor value relative to the move the same agent
would otherwise have made.

## Phase robustness

The selected combined intervention was then tested on 500 fresh pairs at every
starting phase.

| Random opening plies | Paired score | Gain over 0.5 (95% CI) | Decisions changed |
|---:|---:|---:|---:|
| 0 | 0.6395 | +0.1395 [+0.1095, +0.1695] | 51.6% |
| 6 | 0.6540 | +0.1540 [+0.1255, +0.1820] | 52.1% |
| 10 | 0.6675 | +0.1675 [+0.1395, +0.1975] | 51.8% |
| 20 | **0.7115** | **+0.2115 [+0.1855, +0.2360]** | 49.3% |
| 30 | 0.6855 | +0.1855 [+0.1620, +0.2080] | 46.6% |

The effect survives from the initial board through late-midgame openings. It is
not an artifact of the original 10-ply snapshot distribution.

## What has been established

1. Raw game state predicts wins beyond phase, disc lead, color, and simulated
   player strength.
2. Shallow Potage composition improves prediction on fixed untouched games;
   composing again overfits.
3. Cross-fitted selectors agree on ingredient families but not exact formulas.
4. Averaging fold-selected AM models gives a significant held-out improvement.
5. Acting on the stable mobility/safety vocabulary causes a large paired win
   gain, and the gain generalizes across game phases.

This is enough to make another game scientifically pertinent. Further Reversi
work would mainly measure transfer to stronger or different opponent policies;
it would no longer test the core claim that Potage can expose actionable parts
of a game state.

The causal claim remains scoped to this self-play environment and baseline
policy. It is not a claim that the intervention is optimal Reversi strategy.

## Reproduce

```bash
python examples/reversi_intervention.py \
    --output examples/reversi_intervention_results.json \
    --plot examples/reversi_intervention_results.png

python examples/reversi_intervention_phases.py \
    --output examples/reversi_intervention_phases_results.json \
    --plot examples/reversi_intervention_phases_results.png
```
