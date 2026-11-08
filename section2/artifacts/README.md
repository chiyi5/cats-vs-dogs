# section2 可同步产物

本目录只放小体积、可进 Git 的结果文件。

| 文件 | 来源 | 说明 |
|---|---|---|
| `results.csv` | `summarize_results.py` | 每次运行一行：实验、种子、验证准确率、召回、最佳轮次、耗时 |
| `results_summary.csv` | `summarize_results.py` | 按实验汇总的均值、标准差、最小/最大值 |
| `validation_accuracy.png` | `summarize_results.py` | 各次运行的验证准确率柱状图 |
| `learning_curves.png` | `summarize_results.py` | 各次运行的验证准确率曲线 |
| `runs/<实验>_seed<种子>/` | `summarize_results.py`（从 `section2/results/` 复制） | `config.json`、`history.csv`、`metrics.json`、`val_predictions.csv` |
| `submission.csv` | `predict.py` | 500 行 `id,label`，1=dog、0=cat |
| `architecture.png` | `architecture.py` | b) 要求的结构图 |
| `model_summary.txt` | `architecture.py` | 逐层输出维度与参数量 |
| `实验实测报告.md` | 手动撰写 | 结论、因素比较、复现与解释边界 |
| `queue_status.json` | `run_all.py` | 队列执行状态 |

不在此目录（默认被 Git 忽略）：`section2/results/` 下的权重 `best.pt` / `last.pt`、完整日志。

上传前核对：

- `results.csv` 的实验编号与 `plan.md` 一致
- `results_summary.csv` 中 MF 一行有 3 次运行
- `submission.csv` 恰好 500 行
