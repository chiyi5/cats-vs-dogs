"""CIFAR-10 multi-class training (plan.md section 6.2, experiment C1).

Also exposes the shared training loop used by imbalance.py:

  train_and_evaluate(train_view, eval_view, train_indices, val_indices,
                     method, ...) -> metrics dict

The official test set is never used for model selection; it is evaluated once
with the checkpoint chosen on a held-out 5,000-image validation split.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torchvision import datasets, models, transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
VAL_SIZE = 5000
SPLIT_SEED = 42


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight if weight is not None else None)

    def forward(self, logits, targets):
        log_prob = torch.log_softmax(logits, 1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, targets[:, None]).squeeze(1)
        loss = -((1 - prob.gather(1, targets[:, None]).squeeze(1)) ** self.gamma) * target_log_prob
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def cifar_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return train_tf, eval_tf


def load_cifar(root):
    train_tf, eval_tf = cifar_transforms()
    train_view = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    eval_view = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test_view = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    return train_view, eval_view, test_view


def split_indices(total, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """Same 45,000/5,000 split for every experiment so results are comparable."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    val_idx = np.sort(order[:val_size])
    train_idx = np.sort(order[val_size:])
    return train_idx.tolist(), val_idx.tolist()


STEMS = ('imagenet', 'cifar')
EXPERIMENT_STEMS = {'C1': 'imagenet', 'C2': 'cifar'}


def build_model(device, stem='imagenet'):
    """ResNet18 for 32x32 inputs.

    'imagenet' keeps torchvision's 7x7 stride-2 conv + max-pool (C1, H0-H3): a 32x32 image
    shrinks to 8x8 before layer1 and 1x1 at layer4.
    'cifar' follows the CIFAR-10 design of He et al. (2016): a 3x3 stride-1 conv and no
    max-pool, so layer1 sees 32x32 and layer4 sees 4x4 (C2).
    """
    if stem not in STEMS:
        raise ValueError(f'stem must be one of {STEMS}')
    model = models.resnet18(weights=None, num_classes=NUM_CLASSES)
    if stem == 'cifar':
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    return model.to(device)


def class_counts(labels, num_classes=NUM_CLASSES):
    counts = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        counts[label] += 1
    return counts


def train_and_evaluate(train_view, eval_view, train_indices, val_indices, test_view,
                       method, out_dir, seed=42, epochs=30, batch_size=128,
                       workers=2, lr=0.1, amp=False, extra_config=None, stem='imagenet'):
    """Train one CIFAR-10 run and write config/history/metrics into out_dir."""
    for name, value in [('epochs', epochs), ('batch_size', batch_size), ('lr', lr)]:
        if value <= 0:
            raise ValueError(f'{name} must be positive')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'config.json').exists():
        raise FileExistsError(f'{out_dir} already contains a run; choose another --output')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = torch.utils.data.Subset(train_view, train_indices)
    val_dataset = torch.utils.data.Subset(eval_view, val_indices)
    train_labels = [train_view.targets[i] for i in train_indices]
    val_labels = [eval_view.targets[i] for i in val_indices]
    counts = class_counts(train_labels)

    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if method == 'sampler':
        weights = [1.0 / counts[label] for label in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator)

    loss_weight = None
    if method in ('weighted', 'focal'):
        loss_weight = torch.tensor(
            [len(train_labels) / (NUM_CLASSES * max(count, 1)) for count in counts],
            dtype=torch.float32, device=device)
    if method == 'focal':
        loss_fn = FocalLoss(gamma=2.0, weight=loss_weight).to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
            generator=generator if sampler is None else None, num_workers=workers,
            pin_memory=device.type == 'cuda', persistent_workers=workers > 0),
        'val': torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=device.type == 'cuda', persistent_workers=workers > 0),
    }
    model = build_model(device, stem)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type == 'cuda')

    config = {
        'method': method, 'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
        'lr': lr, 'val_size': len(val_indices), 'train_size': len(train_indices),
        'class_counts': counts.tolist(), 'amp': amp, 'workers': workers,
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
        'torch': torch.__version__, 'numpy': np.__version__,
        'split_seed': SPLIT_SEED,
        'protocol': {
            'model': 'resnet18', 'stem': stem, 'weights': 'none (trained from scratch)',
            'optimizer': 'SGD', 'momentum': .9, 'weight_decay': 5e-4,
            'scheduler': 'CosineAnnealingLR', 'loss': method,
            'augmentation': ['RandomCrop(32, padding=4)', 'RandomHorizontalFlip'],
        },
    }
    if extra_config:
        config.update(extra_config)
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))

    best_accuracy, best_epoch = -1.0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        for phase, loader in loaders.items():
            model.train(phase == 'train')
            total_loss, correct, count = 0.0, 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type=device.type,
                                        enabled=amp and device.type == 'cuda'):
                        logits = model(images)
                        loss = loss_fn(logits, labels)
                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
            row[f'{phase}_loss'] = total_loss / count
            row[f'{phase}_accuracy'] = correct / count
            if phase == 'val' and correct / count > best_accuracy:
                best_accuracy, best_epoch = correct / count, epoch
                torch.save(model.state_dict(), out_dir / 'best.pt')
        scheduler.step()
        row['epoch_seconds'] = time.perf_counter() - epoch_start
        history.append(row)
        with (out_dir / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)

    state = torch.load(out_dir / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    predictions, targets = [], []
    test_loader = torch.utils.data.DataLoader(
        test_view, batch_size=batch_size, shuffle=False, num_workers=workers)
    with torch.no_grad():
        for images, labels in test_loader:
            guesses = model(images.to(device)).argmax(1).cpu().tolist()
            predictions.extend(guesses)
            targets.extend(labels.tolist())

    metrics = {
        'experiment': out_dir.name.split('_seed')[0], 'method': method, 'stem': stem, 'seed': seed,
        'test_accuracy': float(np.mean(np.array(predictions) == np.array(targets))),
        'per_class_recall': recall_score(targets, predictions, labels=list(range(NUM_CLASSES)),
                                         average=None, zero_division=0).tolist(),
        'macro_f1': float(f1_score(targets, predictions, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(targets, predictions,
                                             labels=list(range(NUM_CLASSES))).tolist(),
        'best_epoch': best_epoch, 'val_accuracy': best_accuracy,
        'train_count': len(train_indices), 'class_counts': counts.tolist(),
        'elapsed_seconds': time.perf_counter() - start,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # C1: ImageNet stem (baseline). C2: CIFAR stem (3x3 stride-1 conv, no max-pool).
    p.add_argument('--experiment', default='C1', choices=sorted(EXPERIMENT_STEMS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data', type=Path, default=Path('data/cifar10'))
    p.add_argument('--output', type=Path, default=Path('section3/results'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--amp', action='store_true')
    a = p.parse_args()

    a.data.mkdir(parents=True, exist_ok=True)
    train_view, eval_view, test_view = load_cifar(a.data)
    train_idx, val_idx = split_indices(len(train_view))
    train_and_evaluate(train_view, eval_view, train_idx, val_idx, test_view,
                       method='none', out_dir=a.output / f'{a.experiment}_seed{a.seed}',
                       seed=a.seed, epochs=a.epochs, batch_size=a.batch_size,
                       workers=a.workers, lr=a.lr, amp=a.amp,
                       stem=EXPERIMENT_STEMS[a.experiment])


if __name__ == '__main__':
    main()
