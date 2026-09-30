> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Potage stage and selector ablation

This experiment keeps one real-time arena dataset and one game-disjoint
discovery/validation/test split fixed while varying the Potage recipe. Each
recipe is built through the public `SoupPipe` API; no feature matrices are
constructed outside the pipeline.

Recipes:

- `raw`: unary/raw pool, then final selection;
- `raw_am`: raw + amplitude modulation;
- `raw_carriers`: raw + static carriers;
- `deep_am`: raw + AM + a second AM over the selected AM features;
- `carrier_am`: raw + carriers + AM over the carrier features.

Selectors are Potage's `omp`, `greedy`, and `correlation` strategies. The
baseline is the same coarse time/score/skill model used by the real-time probe.

## 600-game matrix

The run produced 595 non-draw games and 10,710 snapshots. Baseline test Brier
was `0.0878`.

| Stage arrangement | OMP | Greedy | Correlation | Best |
|---|---:|---:|---:|---:|
| `raw` | 0.0879 | 0.0886 | 0.0881 | 0.0879 |
| `raw_am` | 0.0877 | 0.0886 | **0.0870** | 0.0870 |
| `raw_carriers` | 0.0868 | 0.0877 | 0.0870 | 0.0868 |
| `deep_am` | 0.0881 | 0.0886 | 0.0875 | 0.0875 |
| `carrier_am` | **0.0866** | 0.0871 | 0.0869 | **0.0866** |

The best single split is `carrier_am + omp`, but the second AM stage is not a
general improvement: `deep_am` is worse than the shallower recipes. Across all
five recipes, correlation has the lowest mean test Brier (`0.0873`), followed by
OMP (`0.0874`) and greedy (`0.0881`).

## Repeated smaller matrix

On three additional 300-game seeds, the best recipe/selector was:

| Seed | Baseline | Best Brier | Best configuration |
|---:|---:|---:|---|
| 19 | 0.0717 | 0.0699 | `raw + correlation` |
| 41 | 0.0911 | 0.0874 | `raw_carriers + correlation` |
| 97 | 0.0957 | 0.0913 | `carrier_am + correlation` |

Correlation improved on the baseline in all three repeats. The exact winning
stage arrangement moves with the sampled games, so the robust conclusion is the
selector choice and the usefulness of shallow carrier/AM composition—not a
single universally optimal deep recipe.

Reproduce the full matrix with:

```bash
python examples/realtime_stage_selection_probe.py \
    --games 600 \
    --output examples/realtime_stage_selection_results.json \
    --plot examples/realtime_stage_selection_results.png
```

Run the smaller repeated check with:

```bash
python examples/realtime_stage_selection_probe.py \
    --games 300 \
    --recipes raw raw_carriers carrier_am \
    --methods omp correlation \
    --stability-seeds 19 41 97 \
    --stability-games 300 \
    --output examples/realtime_stage_selection_stability_results.json
```
