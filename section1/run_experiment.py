"""Controlled Dogs vs. Cats comparisons; run with --help for arguments."""
import argparse
import csv
import json
import random
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from torchvision import datasets, models, transforms as T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Contains train/cat, train/dog, val/cat, val/dog')
    p.add_argument('--experiment', choices=['E1', 'E2', 'E3', 'E4', 'E5'], required=True)
    p.add_argument('--output', type=Path, default=Path('section1/results'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--svm-c', type=float, default=1.0)
    p.add_argument('--amp', action='store_true', help='CUDA mixed precision (apply to all neural comparisons)')
    p.add_argument('--exclude', type=Path, help='JSON list of relative training paths to exclude')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.svm_c <= 0:
        p.error('epochs, batch-size, lr and svm-c must be positive')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    out = a.output / f'{a.experiment}_seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; choose another --output')
    size = 128 if a.experiment == 'E4' else 224
    normalization = T.Normalize([.485, .456, .406], [.229, .224, .225])
    base = [T.Resize(round(size * 256 / 224)), T.CenterCrop(size)]
    aug = [] if a.experiment == 'E2' else [T.RandomHorizontalFlip(), T.RandomRotation(10)]
    train = datasets.ImageFolder(a.data / 'train', T.Compose(base + aug + [T.ToTensor(), normalization]))
    val = datasets.ImageFolder(a.data / 'val', T.Compose(base + [T.ToTensor(), normalization]))
    if a.exclude:
        excluded = set(json.loads(a.exclude.read_text()))
        train.samples = [(path, label) for path, label in train.samples if str(Path(path).relative_to(a.data)) not in excluded]
        train.imgs = train.samples
        train.targets = [label for _, label in train.samples]
    if train.class_to_idx != {'cat': 0, 'dog': 1} or val.class_to_idx != train.class_to_idx:
        raise ValueError('Expected exactly cat and dog directories in both train and val')
    config = {**vars(a), 'size': size, 'class_to_idx': train.class_to_idx,
              'train_count': len(train), 'val_count': len(val),
              'torch': torch.__version__, 'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu'}
    config['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['protocol'] = {'optimizer': 'SGD', 'momentum': .9, 'weight_decay': 1e-4,
                          'scheduler': 'CosineAnnealingLR', 'freeze': False,
                          'weights': 'none' if a.experiment in ['E3', 'E5'] else 'ResNet18_IMAGENET1K_V1',
                          'hog': {'size': 128, 'orientations': 9, 'pixels_per_cell': [16, 16], 'cells_per_block': [2, 2]}}
    (out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    # Save actual file lists to let teammates reproduce and compare splits.
    for name, data in [('train', train), ('val', val)]:
        with (out / f'{name}_files.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['path', 'label'])
            w.writerows((str(Path(path).relative_to(a.data)), label) for path, label in data.samples)
    start = time.perf_counter()
    if a.experiment == 'E5':
        from skimage.feature import hog
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        import joblib

        def features(data):
            result = []
            for path, _ in data.samples:
                with Image.open(path) as image:
                    gray = np.asarray(image.convert('L').resize((128, 128))) / 255.0
                result.append(hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)))
            return np.asarray(result, dtype=np.float32)

        x = features(train)
        xv = features(val)
        classifier = make_pipeline(StandardScaler(copy=False), LinearSVC(C=a.svm_c, random_state=a.seed, max_iter=10000))
        classifier.fit(x, train.targets)
        pred = classifier.predict(xv)
        joblib.dump(classifier, out / 'model.joblib')
        best_epoch = None
        train_seconds = time.perf_counter() - start
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet18(weights=None if a.experiment == 'E3' else models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.to(device)
        scaler = torch.amp.GradScaler('cuda', enabled=a.amp and device.type == 'cuda')
        optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
        loss_fn = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(a.seed)
        loaders = {
            'train': torch.utils.data.DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=generator, num_workers=a.workers, pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0),
            'val': torch.utils.data.DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0),
        }
        best, history, best_epoch = -1, [], None
        for epoch in range(1, a.epochs + 1):
            epoch_start = time.perf_counter()
            row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
            for phase, loader in loaders.items():
                model.train(phase == 'train')
                total_loss, correct, count, predictions = 0., 0, 0, []
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.autocast(device_type=device.type, enabled=a.amp and device.type == 'cuda'):
                            logits = model(images)
                            loss = loss_fn(logits, labels)
                        if phase == 'train':
                            optimizer.zero_grad(set_to_none=True)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    guesses = logits.argmax(1)
                    total_loss += loss.item() * len(labels)
                    correct += (guesses == labels).sum().item()
                    count += len(labels)
                    predictions.extend(guesses.detach().cpu().tolist())
                row[f'{phase}_loss'] = total_loss / count
                row[f'{phase}_accuracy'] = correct / count
                if phase == 'val' and correct / count > best:
                    best, best_epoch, pred = correct / count, epoch, predictions
                    torch.save(model.state_dict(), out / 'best.pt')
            scheduler.step()
            row['epoch_seconds'] = time.perf_counter() - epoch_start
            history.append(row)
            with (out / 'history.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
            print(json.dumps(row), flush=True)
        train_seconds = time.perf_counter() - start
    metrics = {'experiment': a.experiment, 'seed': a.seed,
               'val_accuracy': accuracy_score(val.targets, pred),
               'recall_cat_dog': recall_score(val.targets, pred, labels=[0, 1], average=None, zero_division=0).tolist(),
               'confusion_matrix': confusion_matrix(val.targets, pred, labels=[0, 1]).tolist(),
               'best_epoch': best_epoch, 'elapsed_seconds': train_seconds}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (out / 'val_predictions.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['path', 'label', 'prediction'])
        w.writerows((str(Path(path).relative_to(a.data)), label, int(guess)) for (path, label), guess in zip(val.samples, pred))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
