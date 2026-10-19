"""Rebuild bundled teaching CSVs offline; Python standard library only.

Students do not need Python. Run this from any working directory.
"""
from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parent


def build():
    raw = (ROOT / 'data' / 'auto-mpg.data').read_text().splitlines()
    assert len(raw) == 398, 'Expected the UCI 398-row version.'
    order = list(range(398))
    random.Random(2026).shuffle(order)
    test_rows = set(order[:80])
    rows = []
    for i, line in enumerate(raw):
        fields = line.split()
        # Retain every source row; missing horsepower is not used.
        mpg, displacement = float(fields[0]), float(fields[2])
        weight, acceleration = float(fields[4]), float(fields[5])
        rows.append([format(weight / 1000, '.6g'), format(displacement, '.6g'),
                     format(acceleration, '.6g'), format(mpg, '.6g'),
                     int(i not in test_rows), i + 1])
    with (ROOT / 'data' / 'auto_mpg_lab.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['weight_1000lb', 'displacement_cuin', 'acceleration_s',
                         'mpg', 'is_train', 'source_row'])
        writer.writerows(rows)
    rng = random.Random(17)
    with (ROOT / 'data' / 'synthetic_line.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['x', 'y'])
        for i in range(40):
            x = 10 * i / 39
            y = 3 + 2 * x + rng.gauss(0, 1.5)
            writer.writerow([f'{x:.8f}', f'{y:.8f}'])


if __name__ == '__main__':
    build()
    print('Wrote synthetic_line.csv (40 rows) and auto_mpg_lab.csv (398 rows).')
