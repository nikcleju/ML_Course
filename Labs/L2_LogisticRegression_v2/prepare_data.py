"""Rebuild the lab's teaching CSVs offline (Python standard library only).

The source snapshots are bundled. Students do not need Python.
"""
import csv
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
TITANIC_HEADER = ['sex_code', 'age_scaled', 'fare_scaled', 'survived',
                  'is_train', 'passenger_id']
SPINE_PREDICTORS = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope',
    'pelvic_radius', 'degree_spondylolisthesis', 'pelvic_slope', 'direct_tilt',
    'thoracic_slope', 'cervical_tilt', 'sacrum_angle', 'scoliosis_slope',
]
SPINE_SOURCE_COLUMNS = [f'Col{i}' for i in range(1, 13)]
SPINE_HEADER = SPINE_PREDICTORS + ['abnormal', 'is_train', 'source_row']


def read_source(path):
    """Validate one source partition, rejecting duplicate passenger IDs."""
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'PassengerId', 'Survived', 'Sex', 'Age', 'Fare'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing required source columns.')
        rows = list(reader)
    ids = set()
    for row in rows:
        passenger_id = int(row['PassengerId'])
        if passenger_id <= 0 or passenger_id in ids:
            raise ValueError(f'{path}: invalid or duplicate passenger ID {passenger_id}.')
        ids.add(passenger_id)
        if row['Survived'] not in ('0', '1') or row['Sex'] not in ('0', '1'):
            raise ValueError(f'{path}: target and sex code must be binary.')
        if not all(math.isfinite(float(row[key])) for key in ('Age', 'Fare')):
            raise ValueError(f'{path}: non-finite selected predictor.')
    return rows


def titanic_rows(source_dir):
    train = read_source(source_dir / 'train_data.csv')
    test = read_source(source_dir / 'test_data.csv')
    train_by_id = {row['PassengerId']: row for row in train}
    test_by_id = {row['PassengerId']: row for row in test}
    overlap = train_by_id.keys() & test_by_id.keys()
    for passenger_id in overlap:
        if train_by_id[passenger_id] != test_by_id[passenger_id]:
            raise ValueError(f'Conflicting train/test records for passenger {passenger_id}.')
    if (len(train), len(test), overlap) != (792, 100, {'792'}):
        raise ValueError('Expected the bundled 792/100 source split with shared passenger 792.')
    # Keep the supplied test set unchanged; remove its duplicate from training.
    result = []
    for rows, is_train in ((train, 1), (test, 0)):
        for row in rows:
            if is_train and row['PassengerId'] in overlap:
                continue
            result.append([row['Sex'], row['Age'], row['Fare'], row['Survived'],
                           is_train, row['PassengerId']])
    return result


def spine_rows(source_dir):
    with (source_dir / 'Dataset_spine.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = set(SPINE_SOURCE_COLUMNS) | {'Class_att'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Spine source: missing required columns.')
        rows = list(reader)
    labels = {'Normal': 0, 'Abnormal': 1}
    if len(rows) != 310 or any(row['Class_att'] not in labels for row in rows):
        raise ValueError('Spine source: expected 310 rows with Normal/Abnormal labels.')
    groups = [[i for i, row in enumerate(rows) if labels[row['Class_att']] == label]
              for label in (0, 1)]
    if [len(group) for group in groups] != [100, 210]:
        raise ValueError('Spine source: expected 100 Normal and 210 Abnormal rows.')
    if not all(math.isfinite(float(row[key]))
               for row in rows for key in SPINE_SOURCE_COLUMNS):
        raise ValueError('Spine source: non-finite selected predictor.')
    rng = random.Random(2026)
    test_indices = set()
    for group, count in zip(groups, (20, 42)):
        rng.shuffle(group)
        test_indices.update(group[:count])
    return [[row[key] for key in SPINE_SOURCE_COLUMNS] +
            [labels[row['Class_att']], int(i not in test_indices), i + 1]
            for i, row in enumerate(rows)]


def synthetic_rows():
    rng = random.Random(2026)
    rows = []
    for label, (cx, cy) in enumerate(((-1, 1), (1, -1))):
        for _ in range(60):
            rows.append([f'{rng.gauss(cx, 1):.8f}',
                         f'{rng.gauss(cy, 1):.8f}', label])
    rng.shuffle(rows)
    return rows


def write_csv(path, header, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def build(output_dir=None, source_dir=None):
    """Write prepared CSVs; optional paths support isolated maintenance checks."""
    output_dir = Path(output_dir) if output_dir is not None else ROOT / 'data'
    source_dir = Path(source_dir) if source_dir is not None else ROOT / 'data' / 'source'
    # Validate both sources before creating or replacing any prepared output.
    rows = titanic_rows(source_dir)
    spine = spine_rows(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'titanic_lab.csv', TITANIC_HEADER, rows)
    write_csv(output_dir / 'synthetic_points.csv', ['x1', 'x2', 'y'], synthetic_rows())
    write_csv(output_dir / 'spine_lab.csv', SPINE_HEADER, spine)


if __name__ == '__main__':
    build()
    print('Wrote titanic_lab.csv (791 training + 100 test), synthetic_points.csv '
          '(120 rows), and spine_lab.csv (248 training + 62 test).')
