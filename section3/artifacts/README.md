# section3 可同步产物

| 文件 | 来源 | 说明 |
|---|---|---|
| `error_cases.csv` | `error_analysis.py` | 选出的对/错样例；`error_type` 需人工填写 |
| `error_examples.png` | `error_analysis.py` | 样例拼图，报告 e) 直接引用 |
| `error_analysis.json` | `error_analysis.py` | 验证集总体准确率与选中样例摘要 |
| `results.csv` | `summarize_results.py` | C1 与 H 系列的测试准确率、macro-F1 |
| `per_class_recall.csv` | `summarize_results.py` | 每类召回明细 |
| `per_class_recall.png` | `summarize_results.py` | 类别召回对比柱状图 |
| `confusion_matrices.png` | `summarize_results.py` | H 系列混淆矩阵（按行归一化） |
| `runs/<实验>_seed<种子>/` | `summarize_results.py`（从 `section3/results/` 复制） | config、history、metrics |
| `queue_status.json` | `run_all.py` | 队列状态 |

不在此目录（Git 忽略）：`section3/results/` 下的权重、完整日志；`data/cifar10/` 原始数据。

上传前核对：

- `error_cases.csv` 的 `error_type` 已全部填写
- `results.csv` 包含 C1、H0、H1、H2（如跑了 H3 也应有）
- H 系列的 `train_count` 明显小于 C1，证明不平衡确实生效
