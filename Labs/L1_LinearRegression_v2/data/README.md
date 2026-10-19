# Bundled data and provenance

All required data are present here. Students need no network access or Python.

## Auto MPG

Attribution: Quinlan, R. (1993). *Auto MPG* [Dataset]. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5859H>.

Dataset page: <https://archive.ics.uci.edu/dataset/9/auto+mpg>.
Source data: <https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data>.
Source documentation: <https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names>.
Retrieved 2026-10-19 via the web retrieval tool; `auto-mpg.data` preserves all
398 source records and their fields in source order. Whitespace is a text snapshot,
not a claim of byte-for-byte identity with the remote download.

Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
<https://creativecommons.org/licenses/by/4.0/>. The prepared CSV is an adaptation;
retain this attribution when redistributing it.

`auto_mpg_lab.csv` is built from the 398-row UCI version, not the earlier 406-row
version. UCI had already excluded eight records without MPG. This lab removes
**no additional rows**, including the six records missing horsepower: horsepower
is not selected. No target or selected predictor is missing. No values are imputed.

Exact prepared column order:

1. `weight_1000lb`: original field 5 (vehicle weight in pounds), divided by 1000.
2. `displacement_cuin`: original field 3, engine displacement in cubic inches.
3. `acceleration_s`: original field 6, acceleration-time measurement in seconds.
4. `mpg`: original field 1, city-cycle miles per US gallon; target.
5. `is_train`: split metadata (1 training, 0 test); never a predictor.
6. `source_row`: one-based source record index; never a predictor.

Original cylinders, horsepower, model year, origin, and car name are omitted.
Rows retain the original order. The split is randomized, not a first/last-row
split: `random.Random(2026).shuffle(list(range(398)))` in Python; the first
80 shuffled indices are test rows and the remaining 318 are training rows.
The assignment uses no feature or target values. The stored split is authoritative
for students; MATLAB does not regenerate it. This is a teaching holdout, not a
claim of independence between related car models or representativeness of modern cars.

No normalization is applied. Do not repeatedly tune features against this test set.

## Synthetic line

`synthetic_line.csv` contains 40 rows, columns `x,y`, in artificial units.
Generated for this lab: `x = 10*i/39` for `i=0,...,39`, and
`y = 3 + 2*x + noise`, with independent Gaussian noise of mean 0 and standard
deviation 1.5 using Python `random.Random(17).gauss(0,1.5)`.
CSV values are rounded to eight decimal places. The fitted coefficients will not
exactly equal the generating coefficients. This dataset is for illustration,
not for a generalization claim; it has no train/test split.

## Offline regeneration

From any directory, run `python /path/to/prepare_data.py` (Python 3, standard
library only). This rebuilds both CSVs using the bundled source snapshot.
The raw data are never downloaded by that script. Keep the CSVs with the lab;
regeneration is an instructor maintenance step, not a student prerequisite.
