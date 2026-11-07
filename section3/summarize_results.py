"""Aggregate section3 runs into CSV tables and figures for the report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
FIELDS = ['experiment', 'method', 'seed', 'test_accuracy', 'macro_f1', 'val_accuracy',
          'best_epoch', 'train_count', 'elapsed_seconds']


def load_runs(runs_dir):
    runs = []
    for path in sorted(runs_dir.glob('*/metrics.json')):
        metrics = json.loads(path.read_text())
        metrics['_dir'] = path.parent
        runs.append(metrics)
    return runs


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def copy_small_files(runs, artifacts):
    """Keep git-friendly config/history/metrics in artifacts/runs; weights stay in results/."""
    for run in runs:
        dest = artifacts / 'runs' / run['_dir'].name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'metrics.json', 'history.csv'):
            source = run['_dir'] / name
            if source.exists():
                (dest / name).write_bytes(source.read_bytes())


def plot_recall(runs, destination):
    selected = [run for run in runs if run['experiment'] in ('H0', 'H1', 'H2', 'H3')]
    if not selected:
        return
    selected.sort(key=lambda run: run['experiment'])
    width = 0.8 / len(selected)
    positions = np.arange(len(CIFAR_CLASSES))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for index, run in enumerate(selected):
        axis.bar(positions + index * width, run['per_class_recall'], width,
                 label=f"{run['experiment']} ({run['method']})")
    axis.set_xticks(positions + width * (len(selected) - 1) / 2)
    axis.set_xticklabels(CIFAR_CLASSES, rotation=30, ha='right')
    axis.set_ylabel('Per-class recall')
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_confusions(runs, destination):
    selected = [run for run in runs if run['experiment'] in ('H0', 'H1', 'H2', 'H3')]
    if not selected:
        return
    selected.sort(key=lambda run: run['experiment'])
    figure, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 4))
    axes = [axes] if len(selected) == 1 else axes.flatten()
    for axis, run in zip(axes, selected):
        matrix = np.array(run['confusion_matrix'], dtype=float)
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        axis.set_title(f"{run['experiment']} ({run['method']})", fontsize=10)
        axis.set_xticks(range(len(CIFAR_CLASSES)))
        axis.set_yticks(range(len(CIFAR_CLASSES)))
        axis.set_xticklabels(CIFAR_CLASSES, rotation=90, fontsize=6)
        axis.set_yticklabels(CIFAR_CLASSES, fontsize=6)
    figure.colorbar(image, ax=axes, fraction=0.02)
    figure.savefig(destination, dpi=150, bbox_inches='tight')
    plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('section3/results'))
    p.add_argument('--output', type=Path, default=Path('section3/artifacts'))
    a = p.parse_args()

    runs = load_runs(a.runs)
    if not runs:
        raise SystemExit(f'no metrics.json under {a.runs}; run section3/run_all.py first')

    copy_small_files(runs, a.output)
    write_csv(a.output / 'results.csv', runs, FIELDS)
    recall_rows = []
    for run in runs:
        for index, value in enumerate(run['per_class_recall']):
            recall_rows.append({'experiment': run['experiment'], 'seed': run['seed'],
                                'class': CIFAR_CLASSES[index], 'recall': value})
    write_csv(a.output / 'per_class_recall.csv', recall_rows,
              ['experiment', 'seed', 'class', 'recall'])

    plot_recall(runs, a.output / 'per_class_recall.png')
    plot_confusions(runs, a.output / 'confusion_matrices.png')
    for run in runs:
        print(f"{run['experiment']} ({run['method']}): test_acc={run['test_accuracy']:.4f} "
              f"macro_f1={run['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
