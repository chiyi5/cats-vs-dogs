# section2 — 主模型包（数据 5% + b 15% + c 5% + d 15%）

狗猫二分类的最终模型、超参实验与提交文件。规划见仓库根目录 `plan.md` 第 5 节。

## 依赖

```bash
conda activate ee6483     # PyTorch 2.5.1 / torchvision 0.20.1
```

## 数据

放在 `data/datasets/`（不进 Git），结构：

```
data/datasets/
├── train/{cat,dog}/      # 各 10000 张
├── val/{cat,dog}/        # 各 2500 张
└── test/1.jpg ... 500.jpg
```

训练沿用 `section1/artifacts/excluded_train.json` 的 6 张排除清单，清理后为 19994 张。

## 快速开始

```bash
# 1. 校验数据布局（数量、类别目录、test id 连续）
python section2/prepare_data.py --data data/datasets

# 2. 一致性校验：M0 必须复现 section1 E1 的 99.16%（±0.5pp）
python section2/train.py --data data/datasets --experiment M0 --seed 42 \
    --output section2/results --amp

# 3. 全部实验（M0–M4 + MF 三个种子）
python section2/run_all.py --data data/datasets --amp

# 4. 汇总结果与图表
python section2/summarize_results.py

# 5. 生成提交文件
python section2/predict.py --run section2/results/MF_seed42 \
    --data data/datasets --output section2/artifacts/submission.csv
```

## 实验

| 编号 | 变量 | 种子 |
|---|---|---|
| M0 | 与 E1 完全一致（一致性校验） | 42 |
| M1 | 学习率 1e-4 | 42 |
| M2 | 学习率 1e-2 | 42 |
| M3 | 冻结主干 | 42 |
| M4 | 30 轮 | 42 |
| MF | 最终配置 | 42 / 123 / 2026 |

## 产物

- `artifacts/results.csv`：每次运行的验证准确率
- `artifacts/results_summary.csv`：按实验汇总的均值与标准差
- `artifacts/runs/<实验>_seed<种子>/`：从 `results/` 复制的 config、history、metrics
- `artifacts/submission.csv`：500 行 `id,label`
- `results/`：权重与日志，不进 Git

## 注意

- `M0` 与 `E1` 的差距超过 0.5pp 时，先排查预处理，不要继续后面的实验。
- 权重文件 `*.pt` 被 `.gitignore` 排除，交接到 section3 时用 scp 或集群共享目录。
