"""Aggregate section2 runs into results.csv, a mean/std summary, and figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ['experiment', 'seed', 'validation_accuracy', 'cat_recall', 'dog_recall',
          'best_epoch', 'elapsed_seconds', 'train_count', 'val_count']


def load_runs(runs_dir):
    runs = []
    for metrics_path in sorted(runs_dir.glob('*/metrics.json')):
        metrics = json.loads(metrics_path.read_text())
        recalls = metrics.get('recall_cat_dog', [None, None])
        runs.append({
            'experiment': metrics['experiment'], 'seed': metrics['seed'],
            'validation_accuracy': metrics['val_accuracy'],
            'cat_recall': recalls[0], 'dog_recall': recalls[1],
            'best_epoch': metrics.get('best_epoch'),
            'elapsed_seconds': metrics.get('elapsed_seconds'),
            'train_count': metrics.get('train_count'),
            'val_count': metrics.get('val_count'),
            '_dir': metrics_path.parent,
        })
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


def plot_accuracy(runs, destination):
    labels = [f"{r['experiment']}@{r['seed']}" for r in runs]
    values = [r['validation_accuracy'] * 100 for r in runs]
    figure, axis = plt.subplots(figsize=(max(6, 0.6 * len(runs)), 4))
    axis.bar(labels, values, color='#4c72b0')
    axis.set_ylabel('Validation accuracy (%)')
    axis.set_ylim(max(0, min(values) - 3), 100)
    axis.tick_params(axis='x', rotation=45)
    for index, value in enumerate(values):
        axis.text(index, value + 0.15, f'{value:.2f}', ha='center', fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_curves(runs, destination):
    figure, axis = plt.subplots(figsize=(7, 4))
    for run in runs:
        history = run['_dir'] / 'history.csv'
        if not history.exists():
            continue
        with history.open() as f:
            rows = list(csv.DictReader(f))
        axis.plot([int(r['epoch']) for r in rows],
                  [float(r['val_accuracy']) * 100 for r in rows],
                  label=f"{run['experiment']}@{run['seed']}")
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Validation accuracy (%)')
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('section2/results'))
    p.add_argument('--output', type=Path, default=Path('section2/artifacts'))
    a = p.parse_args()

    runs = load_runs(a.runs)
    if not runs:
        raise SystemExit(f'no metrics.json under {a.runs}; run train.py first')

    copy_small_files(runs, a.output)
    write_csv(a.output / 'results.csv', runs, FIELDS)

    summary = {}
    for run in runs:
        summary.setdefault(run['experiment'], []).append(run['validation_accuracy'])
    summary_rows = []
    for experiment, values in sorted(summary.items()):
        summary_rows.append({
            'experiment': experiment, 'n_runs': len(values),
            'mean_accuracy': float(np.mean(values)),
            'std_accuracy': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'min_accuracy': float(np.min(values)), 'max_accuracy': float(np.max(values)),
        })
    write_csv(a.output / 'results_summary.csv', summary_rows,
              ['experiment', 'n_runs', 'mean_accuracy', 'std_accuracy',
               'min_accuracy', 'max_accuracy'])

    plot_accuracy(runs, a.output / 'validation_accuracy.png')
    plot_curves(runs, a.output / 'learning_curves.png')
    print(json.dumps(summary_rows, indent=2))


if __name__ == '__main__':
    main()
