"""Class-imbalance experiments on CIFAR-10 (plan.md section 6.2, H0-H3).

H0  imbalanced data, no correction (control)
H1  imbalanced data + weighted random sampler (oversample minority)
H2  imbalanced data + class-weighted cross entropy
H3  imbalanced data + focal loss

The validation and test sets stay balanced so per-class recall reflects a real
gain rather than a re-weighted metric.
"""
import argparse
from pathlib import Path

import numpy as np

from cifar import NUM_CLASSES, load_cifar, split_indices, train_and_evaluate

METHODS = {'H0': 'none', 'H1': 'sampler', 'H2': 'weighted', 'H3': 'focal'}
DEFAULT_MINORITY = '0,1,2,3,4'


def imbalanced_positions(targets, minority_classes, fraction, seed):
    """Keep `fraction` of each minority class, all of every other class."""
    rng = np.random.default_rng(seed)
    kept = []
    for cls in sorted(set(targets)):
        positions = [i for i, t in enumerate(targets) if t == cls]
        if cls in minority_classes:
            count = max(1, int(round(len(positions) * fraction)))
            positions = sorted(rng.choice(positions, size=count, replace=False).tolist())
        kept.extend(positions)
    return sorted(kept)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', choices=sorted(METHODS), required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data', type=Path, default=Path('data/cifar10'))
    p.add_argument('--output', type=Path, default=Path('section3/results'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--minority-fraction', type=float, default=0.1)
    p.add_argument('--minority-classes', default=DEFAULT_MINORITY)
    a = p.parse_args()

    if not 0 < a.minority_fraction <= 1:
        p.error('--minority-fraction must be in (0, 1]')
    minority = {int(item) for item in a.minority_classes.split(',') if item.strip()}
    if not minority or any(not 0 <= cls < NUM_CLASSES for cls in minority):
        p.error(f'--minority-classes must be a subset of 0..{NUM_CLASSES - 1}')

    a.data.mkdir(parents=True, exist_ok=True)
    train_view, eval_view, test_view = load_cifar(a.data)
    pool_idx, val_idx = split_indices(len(train_view))
    pool_targets = [train_view.targets[i] for i in pool_idx]

    positions = imbalanced_positions(pool_targets, minority, a.minority_fraction, a.seed)
    train_idx = [pool_idx[position] for position in positions]

    extra_config = {
        'imbalance': {
            'minority_classes': sorted(minority),
            'minority_fraction': a.minority_fraction,
            'pool_size': len(pool_idx),
            'kept_size': len(train_idx),
            'subsample_seed': a.seed,
        },
    }
    train_and_evaluate(train_view, eval_view, train_idx, val_idx, test_view,
                       method=METHODS[a.experiment],
                       out_dir=a.output / f'{a.experiment}_seed{a.seed}',
                       seed=a.seed, epochs=a.epochs, batch_size=a.batch_size,
                       workers=a.workers, lr=a.lr, amp=a.amp, extra_config=extra_config)


if __name__ == '__main__':
    main()
