# AM replay audit: data provenance and availability

The [audit](README.md) can run on concrete alone. Metallic glass is an optional
comparison using an existing local research table. This page separates the
inputs a reader can fetch from those whose upstream provenance is incomplete.

## Concrete: independently downloadable

The audit uses the same loader as the [concrete-strength benchmark](../../examples/concrete_strength.py):
1,030 rows, eight input columns, and compressive strength in MPa from the
[UCI Concrete Compressive Strength dataset](https://doi.org/10.24432/C5PK67),
fetched through OpenML and cached under `--concrete-data-home`.
The audit groups identical seven-ingredient recipes so different curing ages
of one recipe stay in the same split.

The [default reproduction command](README.md#reproduce) runs this comparison
without the metallic-glass CSV or the separate `signalfault` project.

## Metallic glass: exact table not distributed

The saved results used `data/metallic-glass/features.csv` from the author's
separate `signalfault` research checkout. The audit reads the CSV directly;
it does not import or train the neural cascade from that project.

The table contains 843 rows, 81 numeric descriptors, and the target `Tg`
(glass transition temperature, kelvin). Identical descriptor rows form 841
groups. Additional columns `mag_fraction`, `alloy`, and `base_element` are
excluded from the audit's feature matrix. The features are composition
summaries, not the later enriched Magpie/MEGNet representation.

Neither this exact CSV nor its source workbooks are included in Potage or its
release assets. **The metallic-glass results cannot currently be reproduced
from the Potage repository alone.** Their saved metrics, selected expressions,
split hashes, and controls remain available in [results.json](results.json).

### Local preparation record

The local preparation script is named
`research/meta-materials/metallic-glass/build_features.py` in `signalfault`.
It records the following preparation:

1. Read composition, fraction, alloy metadata, and `Tg` from
   `Final_dataset_all_2.xlsx`; retain rows with nonmissing `Tg` and usable
   element/property mappings.
2. Read eight elemental properties from `properties.xlsx`: atomic size,
   melting temperature, Pauling electronegativity, valence electron count,
   Young's modulus, bulk modulus, density, and atomic mass. Missing property
   values are filled with zero. The script corrects carbon's atomic mass from
   `811.0` to `12.011`.
3. Normalize the composition fractions and compute ten statistics per property:
   mean, weighted mean, geometric mean, weighted geometric mean, entropy,
   weighted entropy, range, weighted range, standard deviation, and weighted
   standard deviation. Add `number_of_elements` for 81 descriptors.
4. Write those descriptors followed by `Tg`, `mag_fraction`, `alloy`, and
   `base_element`, preserving retained input-row order.

For the weighted-entropy variant, the local implementation uses
`-sum(fraction * p * log(p))`, where `p` is the normalized absolute property
value plus a small numerical stabilizer. It is not entropy of the normalized
fraction-weighted property distribution. Using another feature generator can
therefore produce different values even with similar column names.

The ordered property names are `atomic_size`, `Tm`, `electronegativity`, `VEC`,
`Youngs_modulus`, `Bulk_modulus`, `Density`, and `atomic_mass`. For each property,
the ordered column prefixes are `mean`, `wtd_mean`, `gmean`, `wtd_gmean`,
`entropy`, `wtd_entropy`, `range`, `wtd_range`, `std`, and `wtd_std`.
`number_of_elements` is first.

### Upstream attribution remains unresolved

Older local research notes attribute the data to
[ZHOU-Ziqing/GAN_BMG](https://github.com/ZHOU-Ziqing/GAN_BMG) and describe it as
UCI-derived. This is a historical attribution, not a verified origin for the
exact two workbooks above.

The repository's [dataset archive at commit `2d486450`](https://github.com/ZHOU-Ziqing/GAN_BMG/blob/2d4864503c517ca9dab52ad63964b977f96b63f0/dataset.zip)
was inspected on 2026-09-30. It contains alloy-family workbooks and GAN-training
files, but neither `Final_dataset_all_2.xlsx` nor `properties.xlsx`. That archive
is therefore **not an established download route for this audit's input**.
The original workbook publication, version, and dataset terms still need to
be identified before offering an exact public reconstruction or redistribution.

### Verified local file identities

These SHA-256 values identify the actual local inputs. The CSV hash matches
the `source_file_sha256` saved with the metallic-glass audit results.

```text
0051284814a41f281f2635d1421551802624e346e0af9a35548b5c7b3b6a4f55  Final_dataset_all_2.xlsx
41c9286e37236ec0af6f8996a95b5b019235e30d3d16d683ef96e04c20cb289a  properties.xlsx
111386a26f110e947e358b020d81679c0542c35a2f32050a150fdd9d574f01a5  features.csv
```

To check an existing CSV without additional tools:

```bash
python -c "import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" /path/to/features.csv
```

The audit also records a SHA-256 over the parsed numeric feature and target
arrays (`data_sha256`), which identifies the numerical input separately from
CSV serialization. Column order and row order matter for matching the recorded
run. Reusing the schema with a different table is a new experiment, not an
exact reproduction.
