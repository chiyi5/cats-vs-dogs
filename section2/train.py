"""Train the main Dogs vs. Cats classifier; see plan.md section 5.

Experiments:
  M0  baseline, identical to section1 E1 (parity check, must match within 0.5pp)
  M1  learning rate 1e-4
  M2  learning rate 1e-2
  M3  frozen backbone, only the classifier head is trained
  M4  30 epochs instead of 20
  MF  final model; defaults to M0 and accepts --lr/--epochs overrides
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, models, transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

EXPERIMENTS = {
    'M0': {'lr': 1e-3, 'epochs': 20, 'freeze': False},
    'M1': {'lr': 1e-4, 'epochs': 20, 'freeze': False},
    'M2': {'lr': 1e-2, 'epochs': 20, 'freeze': False},
    'M3': {'lr': 1e-3, 'epochs': 20, 'freeze': True},
    'M4': {'lr': 1e-3, 'epochs': 30, 'freeze': False},
    'MF': {'lr': 1e-3, 'epochs': 20, 'freeze': False},
}
SIZE = 224
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def build_datasets(data_root, exclude_path, size=SIZE):
    base = [T.Resize(round(size * 256 / 224)), T.CenterCrop(size)]
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(10)]
    train = datasets.ImageFolder(
        data_root / 'train', T.Compose(base + aug + [T.ToTensor(), T.Normalize(MEAN, STD)]))
    val = datasets.ImageFolder(
        data_root / 'val', T.Compose(base + [T.ToTensor(), T.Normalize(MEAN, STD)]))
    if exclude_path and exclude_path.exists():
        excluded = set(json.loads(exclude_path.read_text()))
        train.samples = [(p, y) for p, y in train.samples
                         if str(Path(p).relative_to(data_root)) not in excluded]
        train.imgs = train.samples
        train.targets = [y for _, y in train.samples]
    if train.class_to_idx != {'cat': 0, 'dog': 1} or val.class_to_idx != train.class_to_idx:
        raise ValueError('Expected exactly cat and dog directories in train/ and val/')
    return train, val


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data/datasets'))
    p.add_argument('--experiment', choices=sorted(EXPERIMENTS), required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('section2/results'))
    p.add_argument('--exclude', type=Path,
                   default=Path('section1/artifacts/excluded_train.json'))
    p.add_argument('--epochs', type=int, help='override the experiment default')
    p.add_argument('--lr', type=float, help='override the experiment default')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--amp', action='store_true', help='CUDA mixed precision')
    a = p.parse_args()

    preset = dict(EXPERIMENTS[a.experiment])
    if a.epochs is not None:
        preset['epochs'] = a.epochs
    if a.lr is not None:
        preset['lr'] = a.lr
    if preset['epochs'] < 1 or preset['lr'] <= 0 or a.batch_size < 1:
        p.error('epochs, lr and batch-size must be positive')

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out = a.output / f"{a.experiment}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise FileExistsError(f'{out} already contains a run; choose another --output')

    train_set, val_set = build_datasets(a.data, a.exclude)
    present = sorted(train_set.classes)
    config = {
        **vars(a), **preset, 'size': SIZE,
        'class_to_idx': train_set.class_to_idx, 'classes_present': present,
        'train_count': len(train_set), 'val_count': len(val_set),
        'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
        'numpy': np.__version__,
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol': {
            'model': 'resnet18', 'weights': 'ResNet18_IMAGENET1K_V1',
            'optimizer': 'SGD', 'momentum': .9, 'weight_decay': 1e-4,
            'scheduler': 'CosineAnnealingLR', 'loss': 'CrossEntropyLoss',
            'augmentation': ['RandomHorizontalFlip', 'RandomRotation(10)'],
        },
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    for name, dataset in [('train', train_set), ('val', val_set)]:
        with (out / f'{name}_files.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'label'])
            w.writerows((str(Path(path).relative_to(a.data)), label)
                        for path, label in dataset.samples)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if preset['freeze']:
        freeze_backbone(model)
    model.to(device)

    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.SGD(params, lr=preset['lr'], momentum=.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=preset['epochs'])
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp and device.type == 'cuda')
    generator = torch.Generator().manual_seed(a.seed)
    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set, batch_size=a.batch_size, shuffle=True, generator=generator,
            num_workers=a.workers, pin_memory=device.type == 'cuda',
            persistent_workers=a.workers > 0),
        'val': torch.utils.data.DataLoader(
            val_set, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
            pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0),
    }

    best_accuracy, best_epoch, best_pred, best_prob = -1.0, 0, [], []
    history = []
    start = time.perf_counter()
    for epoch in range(1, preset['epochs'] + 1):
        epoch_start = time.perf_counter()
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        for phase, loader in loaders.items():
            model.train(phase == 'train')
            total_loss, correct, count = 0.0, 0, 0
            predictions, probabilities = [], []
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type=device.type,
                                        enabled=a.amp and device.type == 'cuda'):
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
                probabilities.extend(torch.softmax(logits.float(), 1)[:, 1].detach().cpu().tolist())
            row[f'{phase}_loss'] = total_loss / count
            row[f'{phase}_accuracy'] = correct / count
            if phase == 'val' and correct / count > best_accuracy:
                best_accuracy, best_epoch = correct / count, epoch
                best_pred = list(predictions)
                best_prob = list(probabilities)
                torch.save(model.state_dict(), out / 'best.pt')
        scheduler.step()
        row['epoch_seconds'] = time.perf_counter() - epoch_start
        history.append(row)
        with (out / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)

    torch.save(model.state_dict(), out / 'last.pt')
    metrics = {
        'experiment': a.experiment, 'seed': a.seed,
        'val_accuracy': accuracy_score(val_set.targets, best_pred),
        'recall_cat_dog': recall_score(val_set.targets, best_pred, labels=[0, 1],
                                       average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(val_set.targets, best_pred, labels=[0, 1]).tolist(),
        'best_epoch': best_epoch, 'elapsed_seconds': time.perf_counter() - start,
        'train_count': len(train_set), 'val_count': len(val_set),
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (out / 'val_predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label', 'prediction', 'prob_dog'])
        w.writerows((str(Path(path).relative_to(a.data)), label, int(guess), round(prob, 6))
                    for (path, label), guess, prob in zip(val_set.samples, best_pred, best_prob))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
