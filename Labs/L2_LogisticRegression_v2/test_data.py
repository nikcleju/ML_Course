"""Instructor-only integrity tests; Python standard library, no downloads."""
import csv
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lab2_prepare', ROOT / 'prepare_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class DataTests(unittest.TestCase):
    def test_titanic_schema_and_values(self):
        header, rows = read_csv(ROOT / 'data' / 'titanic_lab.csv')
        self.assertEqual(header, prepare.TITANIC_HEADER)
        self.assertEqual(len(rows), 891)
        self.assertTrue(all(math.isfinite(float(value)) for row in rows for value in row.values()))
        for key in ('sex_code', 'survived', 'is_train'):
            self.assertEqual({row[key] for row in rows}, {'0', '1'})

    def test_fixed_disjoint_split(self):
        _, rows = read_csv(ROOT / 'data' / 'titanic_lab.csv')
        train = [row for row in rows if row['is_train'] == '1']
        test = [row for row in rows if row['is_train'] == '0']
        train_ids = {row['passenger_id'] for row in train}
        test_ids = {row['passenger_id'] for row in test}
        self.assertEqual((len(train), len(test)), (791, 100))
        self.assertEqual(len(train_ids | test_ids), 891)
        self.assertFalse(train_ids & test_ids)
        self.assertIn('792', test_ids)
        for partition in (train, test):
            self.assertEqual({row['survived'] for row in partition}, {'0', '1'})

    def test_source_mapping_and_duplicate_repair(self):
        _, train = read_csv(ROOT / 'data' / 'source' / 'train_data.csv')
        _, test = read_csv(ROOT / 'data' / 'source' / 'test_data.csv')
        train_by_id = {row['PassengerId']: row for row in train}
        test_by_id = {row['PassengerId']: row for row in test}
        self.assertEqual(train_by_id.keys() & test_by_id.keys(), {'792'})
        self.assertEqual(train_by_id['792'], test_by_id['792'])
        _, prepared = read_csv(ROOT / 'data' / 'titanic_lab.csv')
        expected = [(row, '1') for row in train if row['PassengerId'] != '792']
        expected += [(row, '0') for row in test]
        self.assertEqual(len(expected), len(prepared))
        for actual, (source, split) in zip(prepared, expected):
            self.assertEqual(actual, dict(zip(prepare.TITANIC_HEADER,
                [source['Sex'], source['Age'], source['Fare'], source['Survived'],
                 split, source['PassengerId']])))

    def test_synthetic_values(self):
        header, rows = read_csv(ROOT / 'data' / 'synthetic_points.csv')
        self.assertEqual(header, ['x1', 'x2', 'y'])
        self.assertEqual(len(rows), 120)
        self.assertEqual([sum(row['y'] == str(label) for row in rows) for label in (0, 1)], [60, 60])
        rng = random.Random(2026)
        expected = []
        for label, (cx, cy) in enumerate(((-1, 1), (1, -1))):
            for _ in range(60):
                expected.append({'x1': f'{rng.gauss(cx, 1):.8f}',
                                 'x2': f'{rng.gauss(cy, 1):.8f}', 'y': str(label)})
        rng.shuffle(expected)
        self.assertEqual(rows, expected)
        self.assertTrue(all(math.isfinite(float(value)) for row in rows for value in row.values()))

    def test_offline_regeneration_is_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'prepared'
            prepare.build(output_dir=output)
            for name in ('titanic_lab.csv', 'synthetic_points.csv', 'spine_lab.csv'):
                self.assertEqual((output / name).read_bytes(), (ROOT / 'data' / name).read_bytes())

    def test_spine_schema_source_mapping_and_values(self):
        source_path = ROOT / 'data' / 'source' / 'Dataset_spine.csv'
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(),
                         'd49ba0011cf84efafc64eff49a3522937724be03bc3fce4cf038b6763d894d02')
        _, source = read_csv(source_path)
        header, rows = read_csv(ROOT / 'data' / 'spine_lab.csv')
        self.assertEqual(header, prepare.SPINE_HEADER)
        self.assertEqual(len(rows), 310)
        self.assertEqual(len(header), 15)
        self.assertTrue(all(math.isfinite(float(value)) for row in rows for value in row.values()))
        for i, (actual, original) in enumerate(zip(rows, source), start=1):
            self.assertEqual([actual[key] for key in header[:12]],
                             [original[key] for key in [f'Col{i}' for i in range(1, 13)]])
            self.assertEqual(actual['abnormal'], str(int(original['Class_att'] == 'Abnormal')))
            self.assertEqual(actual['source_row'], str(i))
        self.assertEqual({row['is_train'] for row in rows}, {'0', '1'})
        self.assertEqual({row['abnormal'] for row in rows}, {'0', '1'})

    def test_spine_fixed_stratified_disjoint_split(self):
        _, rows = read_csv(ROOT / 'data' / 'spine_lab.csv')
        train = [row for row in rows if row['is_train'] == '1']
        test = [row for row in rows if row['is_train'] == '0']
        train_ids = {int(row['source_row']) for row in train}
        test_ids = {int(row['source_row']) for row in test}
        self.assertEqual((len(train), len(test)), (248, 62))
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(train_ids | test_ids, set(range(1, 311)))
        self.assertEqual([sum(row['abnormal'] == str(label) for row in train)
                          for label in (0, 1)], [80, 168])
        self.assertEqual([sum(row['abnormal'] == str(label) for row in test)
                          for label in (0, 1)], [20, 42])
        rng = random.Random(2026)
        expected_test = set()
        for label, count in ((0, 20), (1, 42)):
            group = [i for i, row in enumerate(rows, start=1) if row['abnormal'] == str(label)]
            rng.shuffle(group)
            expected_test.update(group[:count])
        self.assertEqual(test_ids, expected_test)

    def test_spine_last_predictor_must_be_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            shutil.copytree(ROOT / 'data' / 'source', source)
            header, rows = read_csv(source / 'Dataset_spine.csv')
            rows[0]['Col12'] = 'nan'
            with (source / 'Dataset_spine.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'non-finite selected predictor'):
                prepare.build(output_dir=Path(folder) / 'output', source_dir=source)
            self.assertFalse((Path(folder) / 'output').exists())

    def test_invalid_spine_source_does_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            shutil.copytree(ROOT / 'data' / 'source', source)
            header, rows = read_csv(source / 'Dataset_spine.csv')
            rows[0]['Class_att'] = 'Unknown'
            with (source / 'Dataset_spine.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Spine source'):
                prepare.build(output_dir=Path(folder) / 'output', source_dir=source)
            self.assertFalse((Path(folder) / 'output').exists())

    def test_conflicting_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            shutil.copytree(ROOT / 'data' / 'source', source)
            header, rows = read_csv(source / 'test_data.csv')
            rows[0]['Survived'] = '1'
            with (source / 'test_data.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Conflicting train/test'):
                prepare.build(output_dir=Path(folder) / 'output', source_dir=source)
            self.assertFalse((Path(folder) / 'output').exists())

    def test_duplicate_within_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'duplicate.csv'
            header, rows = read_csv(ROOT / 'data' / 'source' / 'test_data.csv')
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, 'duplicate passenger ID'):
                prepare.read_source(path)


if __name__ == '__main__':
    unittest.main()
