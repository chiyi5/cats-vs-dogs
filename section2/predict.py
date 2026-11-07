"""Predict the 500 test images and write submission.csv (id,label; 1=dog, 0=cat).

Reads a run directory produced by train.py (config.json + best.pt), applies the
same deterministic validation transform, and maps test/<id>.jpg to its numeric id.

Example:
  python section2/predict.py --run section2/results/MF_seed42 \
      --data data/datasets --output section2/artifacts/submission.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms as T

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def numeric_key(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True,
                   help='Run directory containing config.json and the checkpoint')
    p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--data', type=Path, default=Path('data/datasets'))
    p.add_argument('--output', type=Path, default=Path('section2/artifacts/submission.csv'))
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()

    config = json.loads((a.run / 'config.json').read_text())
    checkpoint = a.run / a.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f'{checkpoint} not found; run train.py first')
    size = int(config.get('size', 224))
    class_to_idx = config.get('class_to_idx', {'cat': 0, 'dog': 1})
    dog_index = class_to_idx['dog']

    test_dir = a.data / 'test'
    images = sorted((q for q in test_dir.rglob('*') if q.suffix.lower() in IMAGE_SUFFIXES),
                    key=numeric_key)
    if not images:
        raise SystemExit(f'No test images found under {test_dir}')
    if len(images) != 500:
        print(f'WARNING: expected 500 test images, found {len(images)}', flush=True)

    transform = T.Compose([
        T.Resize(round(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for i in range(0, len(images), a.batch_size):
            batch_paths = images[i:i + a.batch_size]
            batch = torch.stack([transform(Image.open(path).convert('RGB'))
                                 for path in batch_paths]).to(device)
            guesses = model(batch).argmax(1).cpu().tolist()
            for path, guess in zip(batch_paths, guesses):
                rows.append((int(path.stem), int(guess == dog_index)))

    rows.sort()
    ids = [row[0] for row in rows]
    if ids != list(range(1, len(rows) + 1)):
        print('WARNING: ids are not the contiguous sequence 1..N', flush=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(rows)
    dogs = sum(label for _, label in rows)
    print(f'wrote {len(rows)} rows to {a.output} (dog={dogs}, cat={len(rows) - dogs})')


if __name__ == '__main__':
    main()
