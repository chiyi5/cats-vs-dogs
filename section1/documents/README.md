# 双格式初稿维护说明

`literature_review_and_experiments.md` 是正文源文件；同名 `.tex` 由 Pandoc 生成，避免分别编辑造成两版内容不一致。正文保留英文，便于并入小组报告。

本次修订加入：真实数据审计和重复排除、正式超参数、五组完整指标、因素差值、训练曲线、耗时定义、单种子及验证选模的局限。实验数字来自 `../artifacts/runs/`，并经原始验证预测核对。文献广泛筛选和研究团队深读仍未完成，正文已明确标注。

重新生成图表和 LaTeX（在仓库根目录）：

```bash
conda activate ee6483
python section1/build_paper.py --pandoc /path/to/pandoc
```

如果 Pandoc 已在 PATH 中，可省略 `--pandoc`。脚本根据保存的正式实验记录生成 `figures/` 下两张图，再将 Markdown 转成完整 LaTeX 文档。生成过程不重新训练模型。

编译时切换到本目录，以正确解析图像相对路径：

```bash
cd section1/documents
pdflatex -interaction=nonstopmode literature_review_and_experiments.tex
pdflatex -interaction=nonstopmode literature_review_and_experiments.tex
```

也可用 `xelatex` 或 `tectonic` 编译。本次已使用 Tectonic 实际编译，生成8页A4 PDF，并检查结果表、混淆矩阵和学习曲线页面；最终日志未发现长行溢出或公式错误。提交/同步时保留 `.md`、`.tex` 和 `figures/`，需要预览时一并保留 PDF；若需重新生成，也保留 `../build_paper.py` 和 `paper-format.lua`。无需上传编译日志和临时辅助文件。
