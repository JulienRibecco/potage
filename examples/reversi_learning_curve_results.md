> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Fixed-test Reversi learning curve

This experiment isolates the effect of training-set size. One pool of 6,000
generated games is split once into 1,163 validation games, 1,163 untouched test
games, and 3,489 available training games. Every point uses a nested prefix of
the same training-game order and is evaluated on exactly the same test states.
Confidence intervals bootstrap complete test games 1,000 times.

Negative `AM − raw` means shallow composition helps. Positive `AM² − AM` means
the second composition depth hurts.

| Training games | Raw Brier | AM Brier | AM² Brier | AM − raw (95% CI) | AM² − AM (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.15565 | **0.15251** | 0.15447 | −0.00314 [−0.00522, −0.00104] | +0.00196 [+0.00076, +0.00320] |
| 500 | 0.15404 | **0.15195** | 0.15298 | −0.00209 [−0.00421, +0.00011] | +0.00102 [+0.00009, +0.00198] |
| 1,000 | 0.15305 | **0.15060** | 0.15203 | −0.00245 [−0.00414, −0.00065] | +0.00143 [+0.00041, +0.00250] |
| 2,000 | 0.15322 | **0.14982** | 0.15160 | −0.00341 [−0.00503, −0.00177] | +0.00179 [+0.00053, +0.00294] |
| 3,489 | 0.15330 | **0.15030** | 0.15170 | −0.00301 [−0.00471, −0.00151] | +0.00140 [+0.00042, +0.00244] |

## Conclusion

More fitting data does not rescue depth two in this regime. Shallow AM beats
raw at every size and is statistically distinguishable from raw at four of
five sizes. AM² is worse than AM at every size, with all five confidence
intervals excluding zero.

This corrects the earlier changing-test comparison, where the depth-two gap
appeared to vanish at 6,000 generated games. Holding the test games fixed makes
the persistent penalty visible.

The likely bottleneck is feature selection rather than coefficient fitting.
The validation set is deliberately fixed while training data grows; a deeper
candidate search has many more chances to exploit validation-specific patterns.
The next controlled question is therefore whether increasing or cross-fitting
the *selection* data stabilizes AM²—not merely adding more coefficient-fitting
rows.

## Feature stability

- Raw selections become stable: six exact features occur at every size, and
  adjacent-set Jaccard similarity rises from 0.60 to 0.91.
- No exact AM or AM² expression occurs at every size. Adjacent AM Jaccard
  similarity ranges from 0.03 to 0.28; AM² is almost entirely disjoint.
- The ingredients are stable despite the formulas: mobility, frontier discs,
  corners/C-squares, available flips, and edges appear in AM selections at all
  five sizes. X-squares appear at four.

Potage is therefore giving a reproducible vocabulary of win-related state
components, but not yet a reproducible symbolic rule. Interpret ingredient
families first; require stronger stability evidence before interpreting an
individual composite.

## Reproduce

```bash
python examples/reversi_learning_curve.py \
    --output examples/reversi_learning_curve_results.json \
    --plot examples/reversi_learning_curve_results.png
```

The JSON contains all metrics, grouped-bootstrap intervals, and selected
features at every sample size.
