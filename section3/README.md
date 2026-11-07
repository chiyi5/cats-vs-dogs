# section3 — 分析包（e 10% + g 10% + h 10%）

错误分析、CIFAR-10 迁移与类别不平衡实验。规划见 `plan.md` 第 6 节。

## 依赖

```bash
conda activate ee6483     # PyTorch 2.5.1 / torchvision 0.20.1
```

## 输入

| 来源 | 用途 |
|---|---|
| `section2/results/MF_seed<种子>/val_predictions.csv` | e) 错误分析 |
| `section2/results/MF_seed<种子>/config.json` | 确认类别映射 |
| CIFAR-10（首次运行自动下载到 `data/cifar10`） | g)、h) |

## 快速开始

```bash
# e) 错误分析（test 集无标签，因此使用验证集）
python section3/error_analysis.py \
    --predictions section2/results/MF_seed42/val_predictions.csv \
    --data data/datasets --output section3/artifacts

# g) + h)：C1 平衡基线、H0 不平衡对照、H1 过采样、H2 类别加权
python section3/run_all.py --data data/cifar10 --amp

# 可选：H3 Focal Loss
python section3/run_all.py --queue H3:42 --data data/cifar10 --amp

# 汇总
python section3/summarize_results.py
```

## 实验

| 编号 | 数据 | 方法 | 目的 |
|---|---|---|---|
| C1 | CIFAR-10 平衡 | 无 | 多分类基线，报告测试准确率 |
| H0 | 类别 0–4 只剩 10% | 无 | 不平衡对照组 |
| H1 | 同 H0 | 加权随机采样 | 过采样少数类 |
| H2 | 同 H0 | 类别加权交叉熵 | 提高少数类损失权重 |
| H3 | 同 H0 | Focal Loss | 可选补充 |

验证集（5000 张）与测试集（10000 张）始终保持平衡，所以指标反映真实提升，而不是被重新加权出来的数字。

## 产物

- `artifacts/error_cases.csv`：选出的样例，`error_type` 一列需人工填写
- `artifacts/error_examples.png`：样例拼图
- `artifacts/results.csv`：各实验测试准确率、macro-F1
- `artifacts/per_class_recall.csv` 与 `.png`：每类召回
- `artifacts/confusion_matrices.png`：H 系列混淆矩阵对比
- `results/`：权重与日志，不进 Git
