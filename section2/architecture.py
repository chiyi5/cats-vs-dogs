"""Render the ResNet18 classifier as a diagram plus a per-layer output table."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGES = [
    ('conv1', 'Conv 7x7 / stride 2'),
    ('maxpool', 'MaxPool 3x3 / stride 2'),
    ('layer1', 'Layer 1: 2 x BasicBlock'),
    ('layer2', 'Layer 2: 2 x BasicBlock'),
    ('layer3', 'Layer 3: 2 x BasicBlock'),
    ('layer4', 'Layer 4: 2 x BasicBlock'),
    ('avgpool', 'AdaptiveAvgPool'),
    ('fc', 'Linear -> 2 classes'),
]
TRACKED = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
           nn.AdaptiveAvgPool2d, nn.Linear)


def trace_outputs(model, input_shape=(1, 3, 224, 224)):
    outputs = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, TRACKED + (nn.Sequential,)):
            def hook(_module, _inputs, out, name=name):
                if isinstance(out, torch.Tensor):
                    outputs[name] = tuple(out.shape[1:])
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for handle in handles:
        handle.remove()
    return outputs


def write_table(model, outputs, path):
    lines = [f'{"layer":<24}{"type":<16}{"output (C,H,W)":<20}params']
    for name, module in model.named_modules():
        if name in outputs and isinstance(module, TRACKED):
            params = sum(p.numel() for p in module.parameters(recurse=False))
            shape = ','.join(str(d) for d in outputs[name])
            lines.append(f'{name:<24}{type(module).__name__:<16}{shape:<20}{params}')
    total = sum(p.numel() for p in model.parameters())
    lines.append(f'\ntotal parameters: {total:,}')
    path.write_text('\n'.join(lines) + '\n')


def draw(outputs, path):
    heights = []
    for name, label in STAGES:
        shape = outputs[name] if name != 'fc' else None
        detail = '2' if name == 'fc' else ','.join(str(d) for d in shape)
        heights.append((label, detail))

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(heights) + 2)
    ax.axis('off')

    top = len(heights) + 1
    boxes = [('Input', '3,224,224')] + heights
    for i, (label, detail) in enumerate(boxes):
        y = top - i
        color = '#dbeafe' if i else '#e5e7eb'
        if i == len(boxes) - 1:
            color = '#dcfce7'
        ax.add_patch(plt.Rectangle((0.15, y - 0.62), 0.7, 0.55,
                                   facecolor=color, edgecolor='#374151'))
        ax.text(0.5, y - 0.27, f'{label}\n({detail})', ha='center', va='center',
                fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate('', xy=(0.5, y - 1.0), xytext=(0.5, y - 0.62),
                        arrowprops={'arrowstyle': '-|>', 'lw': 1.2,
                                    'color': '#374151'})
    ax.set_title('ResNet18 classifier (Dogs vs Cats)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('section2/artifacts'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    model = models.resnet18(weights=None, num_classes=2).eval()
    outputs = trace_outputs(model)
    write_table(model, outputs, a.output / 'model_summary.txt')
    draw(outputs, a.output / 'architecture.png')
    print(f'wrote {a.output / "architecture.png"} and {a.output / "model_summary.txt"}')


if __name__ == '__main__':
    main()
