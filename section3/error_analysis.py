"""Select high-confidence correct and incorrect validation samples (plan.md e).

The test split has no labels, so error analysis is done on the validation set.
Input is val_predictions.csv produced by section2/train.py.

Example:
  python section3/error_analysis.py --predictions section2/results/MF_seed42/val_predictions.csv \
      --data data/datasets --output section3/artifacts
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

CLASS_NAMES = {0: 'cat', 1: 'dog'}


def read_predictions(path):
    with path.open() as f:
        reader = csv.DictReader(f)
        has_probability = 'prob_dog' in (reader.fieldnames or [])
        rows = []
        for row in reader:
            label = int(row['label'])
            prediction = int(row['prediction'])
            probability = float(row['prob_dog']) if has_probability else 0.5
            confidence = probability if prediction == 1 else 1.0 - probability
            rows.append({
                'path': row['path'], 'label': label, 'prediction': prediction,
                'prob_dog': round(probability, 6), 'confidence': round(confidence, 6),
                'correct': label == prediction,
            })
    return rows, has_probability


def pick(rows, correct, count):
    subset = [row for row in rows if row['correct'] is correct]
    subset.sort(key=lambda row: row['confidence'], reverse=True)
    return subset[:count]


def draw_sheet(selected, data_root, destination, columns=3):
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.2 * rows))
    axes = [axes] if rows * columns == 1 else axes.flatten()
    for axis in axes:
        axis.axis('off')
    for axis, row in zip(axes, selected):
        with Image.open(data_root / row['path']) as image:
            axis.imshow(image.convert('RGB'))
        axis.axis('off')
        truth = CLASS_NAMES[row['label']]
        guess = CLASS_NAMES[row['prediction']]
        color = '#2e7d32' if row['correct'] else '#c62828'
        axis.set_title(f"true={truth} pred={guess} p={row['confidence']:.2f}",
                       fontsize=9, color=color)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/datasets'))
    p.add_argument('--output', type=Path, default=Path('section3/artifacts'))
    p.add_argument('--n-correct', type=int, default=2)
    p.add_argument('--n-errors', type=int, default=4)
    a = p.parse_args()

    rows, has_probability = read_predictions(a.predictions)
    if not has_probability:
        print('WARNING: no prob_dog column; confidence falls back to 0.5', flush=True)
    errors = pick(rows, False, a.n_errors)
    correct = pick(rows, True, a.n_correct)
    selected = errors + correct

    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'error_cases.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_label', 'pred_label', 'prob_dog', 'confidence',
                         'correctness', 'error_type'])
        for row in selected:
            writer.writerow([row['path'], row['label'], row['prediction'], row['prob_dog'],
                             row['confidence'], 'correct' if row['correct'] else 'wrong', ''])
    draw_sheet(selected, a.data, a.output / 'error_examples.png')

    total = len(rows)
    wrong = sum(1 for row in rows if not row['correct'])
    summary = {
        'predictions': str(a.predictions), 'val_count': total, 'wrong_count': wrong,
        'val_accuracy': round(1 - wrong / total, 6),
        'selected': [{'path': r['path'], 'true': CLASS_NAMES[r['label']],
                      'pred': CLASS_NAMES[r['prediction']], 'confidence': r['confidence']}
                     for r in selected],
        'note': 'test split has no labels; analysis is performed on the validation set',
        'annotate': 'fill the error_type column in error_cases.csv by hand',
    }
    (a.output / 'error_analysis.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
