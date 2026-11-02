"""Regenerate paper figures and LaTeX from the Markdown source of truth."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pandoc', default=shutil.which('pandoc'), help='Pandoc executable path')
    args = parser.parse_args()
    if not args.pandoc:
        parser.error('Install Pandoc or provide --pandoc /path/to/pandoc')
    section = Path(__file__).resolve().parent
    docs = section / 'documents'
    figures = docs / 'figures'
    figures.mkdir(exist_ok=True)
    runs = section / 'artifacts/runs'
    ids = ['E1', 'E2', 'E3', 'E4', 'E5']
    accuracy = []
    for exp in ids:
        accuracy.append(json.loads((runs / f'{exp}_seed42/metrics.json').read_text())['val_accuracy'] * 100)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.35, 1]})
    axes[0].bar(ids, accuracy, color=['#22577a', '#38a3a5', '#e9a23b', '#7b6dba', '#8c8c8c'])
    axes[0].set(ylabel='Validation accuracy (%)', ylim=(0, 106), title='Five complete single-seed runs')
    for i, value in enumerate(accuracy):
        axes[0].text(i, value + 1, f'{value:.2f}', ha='center', fontsize=9)
    differences = [accuracy[0] - accuracy[j] for j in [1, 2, 3]]
    axes[1].barh(['Augmentation\nE1 - E2', 'Pretraining\nE1 - E3', 'Resolution\nE1 - E4'], differences, color='#22577a')
    axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].set(xlabel='Accuracy difference (percentage points)', xlim=(-1, 7), title='Controlled contrasts')
    axes[1].invert_yaxis()
    for i, value in enumerate(differences):
        axes[1].text(max(value, 0) + .15, i, f'{value:+.2f}', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / 'validation_accuracy.png', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    colors = ['#22577a', '#38a3a5', '#e9a23b', '#7b6dba']
    for exp, color in zip(ids[:4], colors):
        with (runs / f'{exp}_seed42/history.csv').open() as f:
            history = list(csv.DictReader(f))
        epochs = [int(r['epoch']) for r in history]
        for row, measure in enumerate(['accuracy', 'loss']):
            for col, phase in enumerate(['train', 'val']):
                values = [float(r[f'{phase}_{measure}']) for r in history]
                if measure == 'accuracy': values = [v * 100 for v in values]
                ax = axes[row, col]
                ax.plot(epochs, values, '--' if phase == 'train' else '-', label=exp, color=color)
                ax.set_title(f'{phase.capitalize()} {measure}')
                ax.set_ylabel('Accuracy (%)' if measure == 'accuracy' else 'Cross-entropy loss')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
    for ax in axes[1]: ax.set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(figures / 'learning_curves.png', dpi=220)
    plt.close(fig)

    source = docs / 'literature_review_and_experiments.md'
    subprocess.run([args.pandoc, str(source), '--from=markdown+autolink_bare_uris', '--to=latex', '--standalone',
                    '--lua-filter=' + str(docs / 'paper-format.lua'),
                    '--resource-path=' + str(docs), '-o', str(source.with_suffix('.tex'))], check=True, cwd=docs)
    print('Built figures and LaTeX from', source)


if __name__ == '__main__':
    main()
