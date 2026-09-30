> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Reversi win-factor probe

This experiment asks whether Potage can discover useful parts of a game state
without being given a hand-written position evaluator.

## Protocol

- Generate 1,500 complete Reversi games between stochastic players whose skill
  levels vary independently.
- Record up to five board snapshots per game, always from the player-to-move's
  perspective.
- Predict the eventual winner from 22 measurable state quantities: discs,
  legal mobility, corners, edges, frontier discs, potential mobility,
  corner-adjacent squares, available flips, game phase, and parity.
- Give every model the same baseline: phase, disc lead, player-skill difference,
  and color.
- Split by complete game, so snapshots from a game can never cross partitions.
- Select features on discovery and validation games; report the final score on
  289 untouched games. Lower Brier score is better.

## Main result

| Model | Validation Brier | Test Brier | Test ROC-AUC |
|---|---:|---:|---:|
| Obvious-state baseline | 0.1553 | 0.1393 | 0.8830 |
| Potage raw state | 0.1447 | 0.1259 | 0.9031 |
| Potage shallow AM | 0.1406 | **0.1219** | **0.9080** |
| Potage AM composed again | **0.1382** | 0.1222 | 0.9072 |
| Full-state histogram tree | 0.1391 | 0.1203 | 0.9074 |

The state factors reduce test Brier error by 9.7% relative to the baseline;
shallow composition extends that reduction to 12.5%. The shallow Potage model
gets within 0.0017 Brier of the nonlinear tree.

The second composition depth looks best on validation but slips on untouched
games. That is the exact signature expected if extra composition makes it
easier to fit selection noise.

## What it found

The stable *ingredients* are more informative than any single selected formula:

- move options for each player (mobility);
- the maximum or average number of discs currently flippable;
- frontier exposure;
- corners and the risky squares adjacent to corners;
- interactions between those quantities, rather than disc count alone.

For example, `our_c_squares` has a positive raw residual coefficient even
though C-squares are conventionally risky. The composites qualify it with
corner ownership, mobility, and frontier exposure. This is a good warning that
a marginal coefficient is not automatically a causal rule: a corner-adjacent
square changes meaning once its corner is secured.

## Seed stress test

Three additional 1,000-game simulations used completely new games and splits.

| Seed | Baseline | Raw | AM | AM composed again | Tree |
|---:|---:|---:|---:|---:|---:|
| 19 | 0.1606 | 0.1419 | **0.1343** | 0.1349 | 0.1383 |
| 41 | 0.1803 | 0.1738 | 0.1750 | 0.1719 | **0.1688** |
| 97 | 0.1767 | 0.1561 | 0.1559 | **0.1540** | 0.1620 |

Raw state improves on the baseline in every run. Composition depth is not
uniformly helpful: shallow AM beats raw in two of four total runs, and composing
again beats shallow AM in two of four. Exact selected expressions are unstable,
while their ingredient families recur. The evidence therefore supports using
composition as a hypothesis generator, with depth selected strictly on grouped
held-out games rather than trusted by construction.

## Reproduce

```bash
python examples/reversi_probe.py \
    --output examples/reversi_probe_results.json \
    --plot examples/reversi_probe_results.png \
    --stability-seeds 19 41 97
```

Machine-readable metrics and selected expressions are in
`examples/reversi_probe_results.json`.

## More-data check

The identical protocol was also run with 6,000 games (28,933 snapshots and
1,163 untouched test games):

| Model | Validation Brier | Test Brier |
|---|---:|---:|
| Obvious-state baseline | 0.17019 | 0.16260 |
| Potage raw state | 0.15307 | 0.14424 |
| Potage shallow AM | 0.14778 | **0.140018** |
| Potage AM composed again | **0.14752** | **0.140017** |
| Full-state histogram tree | 0.14302 | 0.13505 |

More data removes the visible AM² overfit penalty, supporting the data-starved
hypothesis. It does not reveal an AM² benefit: its test improvement over AM is
0.0000008 Brier, effectively zero. The validation advantage also shrinks from
0.00245 to 0.00026. On present evidence, more data stabilizes the second depth,
but the shallow representation contains nearly all of its predictive value.

The absolute scores should not be compared as a strict learning curve because
the larger grouped split contains a different set of test games. The within-run
AM-versus-AM² comparison is valid. A definitive sample-efficiency curve should
reserve one fixed test-game set and train on nested subsets of the remaining
games.

The larger result is in `examples/reversi_probe_large_results.json`, with its
plot in `examples/reversi_probe_large_results.png`.
