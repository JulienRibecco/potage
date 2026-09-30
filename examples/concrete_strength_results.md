# Concrete strength: corrected reference run


This fixed-seed rerun includes the core refactoring, module separation, and replay/scoring fixes documented in [CHANGELOG.md](../CHANGELOG.md). It supersedes the earlier numerical results and their interpretation. The module-separation rerun reproduced every validation/test metric from the corrected reference within 1e-10.


The UCI Concrete Compressive Strength dataset contains 1,030 measurements and eight inputs ([dataset DOI](https://doi.org/10.24432/C5PK67)). Rows are grouped by the seven ingredient quantities: different curing ages for one recipe remain together.


The split is unchanged: 648 discovery rows (256 recipes), 187 validation rows (86 recipes), and 195 test rows (86 recipes). Feature selection uses validation labels; estimators fit discovery data. This is a diagnostic rerun of an already inspected split, not a fresh independent confirmation.


| Model | Validation R² | Test R² | Test MAE (MPa) |
|---|---:|---:|---:|
| mean | -0.001 | -0.005 | 12.980 |
| raw ridge | 0.552 | 0.652 | 7.504 |
| quadratic ridge | 0.680 | 0.815 | 5.337 |
| hist gradient boosting | 0.841 | 0.894 | 4.012 |
| random forest | 0.804 | 0.878 | 4.316 |
| extra trees | 0.861 | 0.893 | 4.006 |
| stage 0 unary | 0.755 | 0.833 | 5.277 |
| stage 1 am | 0.814 | 0.812 | 5.516 |
| stage 2 carriers | 0.853 | 0.730 | 6.466 |
| stage 3 fm pm | 0.901 | 0.749 | 6.204 |
| stage 4 fm nonsmooth | 0.919 | 0.773 | 5.989 |
| stage 5 wm | 0.918 | 0.792 | 5.591 |

Unary features remain useful for this split, improving raw ridge while tree ensembles perform best. The deeper stages have higher validation scores but lower test scores than the unary checkpoint. These are descriptive comparisons; selecting the unary checkpoint after inspecting test scores is not a validated selection procedure.

The previous AM test R² was 0.418; it is now 0.812. Replay had recalculated normalization and denominator statistics on validation/test batches, while tied ranks differed between fitting and replay. Therefore the earlier attribution of the drop solely to feature-selection overfitting was not justified. The fixes also clip power exponents to their training range; the change in score cannot isolate the contribution of each correction.

Remaining limits: a single split, repeated validation-based feature selection, potentially unstable extrapolation, and fixed baseline settings. RidgeCV uses its default internal validation within discovery data; only the outer discovery/validation/test boundaries are recipe-grouped. No causal or state-of-the-art claim is made.

The [JSON](concrete_strength_results.json) records all scores, feature names, versions, and source hashes. The [figure](concrete_strength_results.png) is regenerated from it.

```bash
pip install -e '.[examples]'
python examples/concrete_strength.py --output artifacts/concrete.json --plot artifacts/concrete.png
```
