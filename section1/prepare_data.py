"""Safely extract the course archive and audit all image files without changing labels."""
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'data'
out = ROOT / 'section1/artifacts'
out.mkdir(exist_ok=True)
archive = data / 'datasets.zip'
with zipfile.ZipFile(archive) as z:
    target = data.resolve()
    for member in z.infolist():
        if not (target / member.filename).resolve().is_relative_to(target):
            raise ValueError('Unsafe archive entry')
    print('Archive entries:', len(z.infolist()), flush=True)
    z.extractall(data)
roots = [p.parent for p in data.rglob('train') if p.is_dir() and (p.parent / 'val').is_dir()]
if len(roots) != 1:
    raise ValueError(f'Expected one dataset root, got {roots}')
root = roots[0]
counts, hashes, bad, modes = Counter(), defaultdict(list), [], Counter()
rows = []
for split in ['train', 'val', 'test']:
    for p in sorted((root / split).rglob('*')):
        if not p.is_file() or p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        rel = str(p.relative_to(root))
        label = p.parent.name if split != 'test' else 'unknown'
        try:
            with Image.open(p) as im:
                im.load()
                width, height, mode = im.width, im.height, im.mode
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[digest].append(rel)
            counts[f'{split}/{label}'] += 1
            modes[mode] += 1
            rows.append([rel, split, label, width, height, mode, digest])
        except Exception as e:
            bad.append({'path': rel, 'error': str(e)})
duplicates = [v for v in hashes.values() if len(v) > 1]
cross = [v for v in duplicates if len({p.split('/')[0] for p in v}) > 1]
report = {'root': str(root), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'counts': dict(counts), 'modes': dict(modes), 'bad_images': bad,
          'exact_duplicate_groups': duplicates, 'cross_split_duplicates': cross,
          'note': 'Exact byte hashes only; perceptual duplicates and label correctness are not established.'}
excluded = []
for group in duplicates:
    training = sorted(p for p in group if p.startswith('train/'))
    other = [p for p in group if not p.startswith('train/')]
    excluded.extend(training if other else training[1:])
(out / 'excluded_train.json').write_text(json.dumps(sorted(excluded), indent=2))
report['excluded_training_files'] = sorted(excluded)
report['clean_train_count'] = sum(v for k, v in counts.items() if k.startswith('train/')) - len(excluded)
(out / 'data_audit.json').write_text(json.dumps(report, indent=2))
with (out / 'data_manifest.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['path', 'split', 'label', 'width', 'height', 'mode', 'sha256']); w.writerows(rows)
print(json.dumps(report, indent=2), flush=True)
if bad:
    raise RuntimeError('Review bad images before training')
