> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Cross-fitted Reversi feature consensus

Five independent selectors were trained on complete-game folds from 3,104
development games. Each selector used its held-out development fold for feature
choice, and every fitted model predicted the same 776-game untouched test set.
Fold predictions were averaged. Confidence intervals bootstrap complete test
games 1,000 times.

## Predictive result

| Model | Test Brier | ROC-AUC |
|---|---:|---:|
| Full-development baseline | 0.17334 | 0.8184 |
| Cross-fitted raw-selection ensemble | 0.15638 | 0.8520 |
| Cross-fitted AM-selection ensemble | **0.15136** | **0.8596** |
| Majority ingredient consensus | 0.15593 | 0.8531 |
| Majority exact-expression consensus | 0.16168 | 0.8398 |

The AM ensemble improves Brier over the raw ensemble by 0.00501, with a
game-bootstrap 95% interval of [0.00326, 0.00683]. The improvement is therefore
not explained by a few lucky test games.

## What is stable

Seven ingredients occur in every fold's selected AM expressions:

- our mobility and mean available flips;
- our frontier discs, corners, and C-squares;
- opponent frontier discs and corners.

Empty-square parity, opponent C-squares and mobility, and our disc count occur
in four of five folds. These form a credible recurring vocabulary for this
self-play distribution.

The 3-of-5 majority ingredient model improves the baseline by 0.01741 Brier
(95% interval [0.01319, 0.02184]) and almost matches the cross-fitted raw
ensemble. That confirms that the recurring ingredients carry real held-out
signal.

## What is not stable

Mean pairwise Jaccard similarity between the exact AM selections is only 0.047.
No exact expression appears in four of five folds. Only two reach a simple
3-of-5 majority:

- `relu(our_mean_flips-opponent_corners)`
- `|our_discs-our_corners|/Σ`

Together they improve the baseline by 0.01166 Brier (95% interval
[0.00754, 0.01560]), but trail both the ingredient model and the AM ensemble.
The symbolic formulas are therefore useful local proxies, not stable laws of
Reversi.

## Interpretation

Cross-fitting resolves the apparent contradiction: Potage's shallow composite
space has reproducible predictive value, while many correlated expressions can
represent that value. Averaging independently selected models preserves the
shared signal and attenuates expression-specific noise. For explanation, report
ingredient recurrence; for prediction, use the cross-fitted ensemble.

The next causal test should intervene on one stable concept—mobility conditioned
on corner/frontier safety—and measure whether an agent deliberately favoring it
wins more often against matched opponents.

## Reproduce

```bash
python examples/reversi_crossfit_probe.py \
    --output examples/reversi_crossfit_results.json \
    --plot examples/reversi_crossfit_results.png
```

The JSON contains fold selections, expression and ingredient frequencies,
metrics, and grouped-bootstrap intervals.
