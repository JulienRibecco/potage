> Historical result: predates replay/scoring fixes; not revalidated for this release.
> See [validation notes](../CHANGELOG.md) before using these scores or conclusions.

# Continuous scenario coverage probe

This probe samples start positions continuously from the playable 5x5 cells and
trains on speed pairs `(1,1)`, `(2,1)`, `(1,2)`, and `(2,2)`. It then evaluates
on a fresh in-support sample and on a held-out speed regime containing speed 3:
`(3,1)`, `(1,3)`, and `(3,3)`. Each model uses the same baseline and
`carrier_am` Potage recipe; only the feature space differs.

## Result

Run: 1,200 training games, seed 1201, 12 selected features.

| Feature space | In-support Brier | Held-out-speed Brier | In-support AUC | Held-out-speed AUC |
|---|---:|---:|---:|---:|
| Core state | 0.0603 | 0.0662 | 0.9769 | 0.9723 |
| Core + scenario | 0.0604 | 0.0672 | 0.9767 | 0.9718 |

Scenario terms were selected, but they did not improve the held-out speed
regime. This differs from the small smoke run, so the apparent gain there is
not stable enough to treat as evidence.

## Interpretation

Continuous coverage of start positions and the known speed range is not enough
by itself. The core trajectory/state representation already captures most of
the predictive signal in this arena; adding raw speed and start descriptors
creates a small selection penalty and slightly worsens transport to speed 3.

The next probe should therefore change the target rather than only add more
descriptors: fit a conditional effect model for the intervention-derived
speed/position effects, or train on the held-out speed range as well and test
whether the selected terms become calibrated. A raw feature win on one split is
not sufficient; the criterion should be repeated held-out regimes and paired
intervention agreement.
