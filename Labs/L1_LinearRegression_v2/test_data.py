"""Instructor-only, standard-library integrity checks; no student dependencies."""
import csv
import importlib.util
import math
from pathlib import Path
import random
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def read_csv(name):
    with (ROOT / 'data' / name).open(newline='') as f:
        reader = csv.reader(f)
        return next(reader), [[float(v) for v in row] for row in reader]


class DataTests(unittest.TestCase):
    def test_car_schema_and_finiteness(self):
        header, rows = read_csv('auto_mpg_lab.csv')
        self.assertEqual(header, ['weight_1000lb', 'displacement_cuin',
                                 'acceleration_s', 'mpg', 'is_train', 'source_row'])
        self.assertEqual(len(rows), 398)
        self.assertTrue(all(len(r) == 6 for r in rows))
        self.assertTrue(all(math.isfinite(v) for r in rows for v in r))
        self.assertTrue(all(v > 0 for r in rows for v in r[:4]))

    def test_source_mapping_and_units(self):
        raw = (ROOT / 'data' / 'auto-mpg.data').read_text().splitlines()
        _, rows = read_csv('auto_mpg_lab.csv')
        self.assertEqual(len(raw), 398)
        self.assertEqual(sum(line.split()[3] == '?' for line in raw), 6)
        for i, (source, row) in enumerate(zip(raw, rows)):
            fields = source.split()
            self.assertAlmostEqual(row[0], float(fields[4]) / 1000, places=8)
            self.assertEqual(row[1:4], [float(fields[2]), float(fields[5]), float(fields[0])])
            self.assertEqual(row[5], i + 1)

    def test_fixed_disjoint_split(self):
        _, rows = read_csv('auto_mpg_lab.csv')
        order = list(range(398))
        random.Random(2026).shuffle(order)
        expected_test = set(order[:80])
        train = {i for i, r in enumerate(rows) if r[4] == 1}
        test = {i for i, r in enumerate(rows) if r[4] == 0}
        self.assertEqual(test, expected_test)
        self.assertEqual((len(train), len(test)), (318, 80))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(range(398)))

    def test_synthetic_values(self):
        header, rows = read_csv('synthetic_line.csv')
        self.assertEqual(header, ['x', 'y'])
        self.assertEqual(len(rows), 40)
        rng = random.Random(17)
        for i, (x, y) in enumerate(rows):
            exact_x = 10 * i / 39
            self.assertAlmostEqual(x, exact_x, places=7)
            self.assertAlmostEqual(y, 3 + 2 * exact_x + rng.gauss(0, 1.5), places=7)

    def test_offline_regeneration_is_identical(self):
        spec = importlib.util.spec_from_file_location('lab1_prepare', ROOT / 'prepare_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            (temp / 'data').mkdir()
            shutil.copyfile(ROOT / 'data' / 'auto-mpg.data', temp / 'data' / 'auto-mpg.data')
            module.ROOT = temp
            module.build()
            for name in ['auto_mpg_lab.csv', 'synthetic_line.csv']:
                self.assertEqual((ROOT / 'data' / name).read_bytes(),
                                 (temp / 'data' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
