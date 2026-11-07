"""Verify the course dataset layout before training; see plan.md section 3.1."""
import argparse
import json
import sys
from pathlib import Path

EXPECTED = {
    'train/cat': 10000,
    'train/dog': 10000,
    'val/cat': 2500,
    'val/dog': 2500,
    'test': 500,
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def count_images(directory):
    return sum(1 for q in directory.rglob('*') if q.suffix.lower() in IMAGE_SUFFIXES)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/datasets'))
    p.add_argument('--exclude', type=Path,
                   default=Path('section1/artifacts/excluded_train.json'))
    p.add_argument('--allow-mismatch', action='store_true',
                   help='report mismatches without failing (smoke tests)')
    a = p.parse_args()

    problems = []
    if not (a.data / 'train').is_dir() or not (a.data / 'val').is_dir():
        raise SystemExit(f'{a.data} does not contain train/ and val/')

    report = {'root': str(a.data.resolve()), 'counts': {}}
    for split in ['train', 'val']:
        for category in sorted(q.name for q in (a.data / split).iterdir() if q.is_dir()):
            actual = count_images(a.data / split / category)
            report['counts'][f'{split}/{category}'] = actual
            expected = EXPECTED.get(f'{split}/{category}')
            if expected is not None and actual != expected:
                problems.append(f'{split}/{category}: expected {expected}, found {actual}')
    test_dir = a.data / 'test'
    test_count = count_images(test_dir) if test_dir.is_dir() else 0
    report['counts']['test'] = test_count
    if not test_dir.is_dir():
        problems.append('test/ directory is missing')
    elif test_count != EXPECTED['test']:
        problems.append(f'test: expected {EXPECTED["test"]}, found {test_count}')

    ids = sorted(int(q.stem) for q in test_dir.rglob('*')
                 if q.suffix.lower() in IMAGE_SUFFIXES and q.stem.isdigit())
    if len(ids) != test_count:
        problems.append('some test images do not have numeric filenames')
    elif ids != list(range(1, test_count + 1)):
        problems.append('test ids are not contiguous 1..N')

    if a.exclude.exists():
        excluded = json.loads(a.exclude.read_text())
        report['excluded_train'] = len(excluded)
        missing = [rel for rel in excluded if not (a.data / rel).exists()]
        if missing:
            problems.append(f'{len(missing)} excluded paths do not exist, e.g. {missing[0]}')
        report['clean_train_count'] = sum(
            report['counts'].get(f'train/{c}', 0) for c in ['cat', 'dog']) - len(excluded)
    else:
        report['excluded_train'] = None
        report['clean_train_count'] = report['counts'].get('train/cat', 0) + \
            report['counts'].get('train/dog', 0)
        problems.append(f'exclusion list not found at {a.exclude}')

    report['problems'] = problems
    print(json.dumps(report, indent=2))
    if problems and not a.allow_mismatch:
        print('FAILED: fix the problems above before training', file=sys.stderr)
        raise SystemExit(1)
    print('OK: dataset layout matches plan.md section 3.1')


if __name__ == '__main__':
    main()
