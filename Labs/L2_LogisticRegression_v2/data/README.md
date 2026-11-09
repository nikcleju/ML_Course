# Bundled data

All student data are local. No download or Python installation is needed.

## Titanic

`titanic_lab.csv` reuses the already transformed Titanic files bundled with
Lab 2. Unchanged snapshots are in `source/train_data.csv` and
`source/test_data.csv`.

The original worksheet attributes these files to Azeem Bootwala's *Titanic*
dataset on Kaggle:
<https://www.kaggle.com/datasets/azeembootwala/titanic>.
This attribution is inherited from that worksheet, not independently verified.
The source snapshots do not include a licence; redistribution rights have not
been established by this revision. Do not infer a data licence from the course's
licence. Confirm the dataset's terms before redistributing outside the existing
course context.

### Columns

| Column | Meaning | Use |
|---|---|---|
| `sex_code` | Supplied binary `Sex` code (0 or 1) | Predictor |
| `age_scaled` | Supplied transformed `Age` value | Predictor |
| `fare_scaled` | Supplied transformed `Fare` value | Predictor |
| `survived` | 0 = did not survive, 1 = survived | Target |
| `is_train` | 1 = training, 0 = test | Split metadata only |
| `passenger_id` | Original `PassengerId` | Identifier only |

The precise sex-code mapping, scaling constants, and upstream missing-value
handling are not documented in the bundled files. Do not guess the mapping or
interpret age/fare values as years/currency. This lab does not add scaling or
imputation, or reverse the existing transformations. No selected value is missing
or non-finite. All other source columns, including the unnamed row index, are
omitted.

### Split repair

The old training snapshot has 792 rows and the test snapshot has 100 rows.
Passenger 792 occurs in both as an identical full record. This revision removes
that row **from training only**, leaving **791 training rows and 100 test rows**:
891 distinct passenger IDs. Other rows and their supplied split assignments are
unchanged, as are all selected predictor/target values. Prepared rows are in
training-source order, followed by test-source order.

The split is inherited, not newly randomized. The upstream preprocessing may
have used the full dataset; its fitting population is unknown. Removing the
shared passenger repairs the known row overlap, but does **not** establish that
all upstream preparation was independent of the test set. Related passengers
may also occur across partitions. Treat this as a controlled teaching comparison,
not a leakage-free benchmark or an estimate for modern populations. Associations
in these historical records do not establish causal effects.

## Synthetic points

`synthetic_points.csv` has 120 rows, columns `x1,x2,y`, in artificial units.
There are 60 observations in each class. Independent Gaussian coordinates use
class centres (-1,1) for class 0 and (1,-1) for class 1, with standard deviation
1 in each coordinate. `random.Random(2026)` generates all coordinates, then
shuffles the rows. Coordinates are rounded to eight decimal places.

The classes overlap; a fitted boundary need not classify every point correctly.
This is a visual training demonstration, not a held-out generalization test.

## Spine: independent exercise

`spine_lab.csv` retains all 12 numeric predictors from the original Lab 2's
`Spine/Dataset_spine.csv`; an unchanged snapshot is bundled as
`source/Dataset_spine.csv`.

The original worksheet attributes the dataset to:
<https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset>.
The page resolves, but its body was not readable through automated retrieval
on 2026-11-09; its description and licence have not been independently verified.
No dataset licence is bundled; confirm redistribution terms before wider distribution.
A readable supplemental source is Raihan-Al-Masud and Mondal (2020),
[Methodology and Figure 1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0228422),
which explicitly describes this Kaggle dataset as 310 records with 12 attributes,
100 Normal and 210 Abnormal labels. The paper's binary coding is opposite to our
lab: our CSV always uses Normal = 0 and Abnormal = 1.

| Column | Source | Use |
|---|---|---|
| `pelvic_incidence` | `Col1` | Predictor |
| `pelvic_tilt` | `Col2` | Predictor |
| `lumbar_lordosis_angle` | `Col3` | Predictor |
| `sacral_slope` | `Col4` | Predictor |
| `pelvic_radius` | `Col5` | Predictor |
| `degree_spondylolisthesis` | `Col6` | Predictor |
| `pelvic_slope` | `Col7` | Predictor |
| `direct_tilt` | `Col8` | Predictor |
| `thoracic_slope` | `Col9` | Predictor |
| `cervical_tilt` | `Col10` | Predictor |
| `sacrum_angle` | `Col11` | Predictor |
| `scoliosis_slope` | `Col12` | Predictor |
| `abnormal` | `Class_att`: Normal = 0, Abnormal = 1 | Target |
| `is_train` | Fixed split: training = 1, test = 0 | Metadata only |
| `source_row` | One-based original row number | Identifier only |

Predictor names come from annotations in the source's unnamed trailing column.
Their measurement units and collection details are not documented in the local
snapshot. Values are preserved exactly; no scaling or imputation is added.
Only the trailing annotation column is omitted. In the prepared CSV, columns
1–12 are predictors, column 13 is the target, column 14 is the split flag, and
column 15 is the source-row identifier.

The supplied pelvic incidence is approximately pelvic tilt plus sacral slope
(maximum absolute residual about 1e-8 in the bundled snapshot). All 12 predictors
are deliberately retained, as requested, but this near dependency makes individual
coefficients numerically unstable. MATLAB may flag conditioning/rank deficiency;
[GeneralizedLinearModel documentation](https://www.mathworks.com/help/stats/generalizedlinearmodel.html)
describes setting redundant coefficients to zero for rank-deficient models.
Whether this rounded near dependency crosses MATLAB's internal threshold has not
been tested natively. Do not interpret coefficient size as predictor importance.

All 310 records remain in their original order: 100 Normal and 210 Abnormal.
The fixed stratified split uses Python `random.Random(2026)`: shuffle the list
of Normal source indices first, then Abnormal indices using the same generator;
reserve the first 20 Normal and first 42 Abnormal indices for testing. This
leaves **248 training rows (80 Normal, 168 Abnormal)** and **62 test rows
(20 Normal, 42 Abnormal)**. No source row occurs in both partitions. Students
use the stored split rather than regenerating it.

This is a teaching classification exercise, **not a medical diagnostic tool**.
Row-disjointness does not establish representativeness or independent patients;
patient identifiers and sampling details are not documented in the snapshot.

## Instructor maintenance

Run `python /path/to/prepare_data.py` from any directory. It rebuilds the three
prepared CSVs using only the bundled snapshots and Python's standard library.
It rejects duplicate IDs within either source partition, conflicting records
across partitions, and an unexpected Titanic source split. Spine input must have 310 finite
selected records with the expected class labels/counts. Both datasets are
validated before any prepared CSV is written. The source snapshots are not
changed or downloaded.

Run checks with:

```sh
python -B -m unittest discover -s /path/to/L2_LogisticRegression_v2 -p test_data.py -v
```
