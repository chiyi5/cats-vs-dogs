# 可同步的实验产物

- `data_audit.json`：课程数据原始数量、图片解码结果与完全重复的文件组。
- `data_manifest.csv`：原始25,500张图片的尺寸、类别、相对路径、SHA256。
- `excluded_train.json`：训练排除清单；6张，不删除原始文件。1张跨训练/验证重复，其余为训练内部重复副本。
- `queue_status.json`：五组正式实验队列状态；running或failed均不表示已完成。
- `results.csv`、`runs/`、图表及`实验实测报告.md`：由已完成实验生成，不填充未完成结果。

正式协议：清理后19994张训练图片、原始5000张验证图片，E1–E4均20轮、batch size16、种子42、2个加载worker、CUDA混合精度。学习率0.001的SGD配余弦调度，全量训练参数。E5为128灰度HOG（16像素单元）+标准化+线性SVM。此轮为单种子受控比较，不是包②最终配置的3种子实验。

原始图片、权重、全量验证预测与运行日志保存在`data/`和`section1/results/`，默认不提交Git。上传前复核队列完成、结果完整、报告中的限制以及数据文件名是否适合共享。尚未执行Git提交或push。

复现：激活ee6483，将教师压缩包置于data/datasets.zip，执行`python section1/prepare_data.py`，再执行`python section1/run_all.py`。队列跳过已有metrics.json的运行；对有config但无metrics的失败运行，先保留到其他路径，再重跑，避免覆盖记录。
