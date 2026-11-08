# EE6483 Mini Project (Option 2): Dogs vs. Cats

Group project for EE6483 Artificial Intelligence and Data Mining. The final report is [`section1/documents/final_report.pdf`](section1/documents/final_report.pdf); the submitted predictions are [`section2/artifacts/submission.csv`](section2/artifacts/submission.csv).

| Folder | Contents | Report sections |
|---|---|---|
| `section1/` | Data audit, duplicate removal, comparative experiments E1–E5, report source | 1, 2, 7 |
| `section2/` | Main model (ResNet18), hyperparameter study M0–M4, multi-seed runs MF, test predictions | 3, 4, 5 |
| `section3/` | CIFAR-10 (C1, C2), class imbalance (H0–H2), error analysis | 6, 8, 9 |

Each `*/artifacts/runs/<experiment>_seed<seed>/` folder holds the `config.json` (settings, software versions, GPU), `history.csv` (per-epoch curves) and `metrics.json` (final results) of one run. Every number in the report comes from these files. Model weights are not stored in the repository.

## 1. Environment

Python 3.10 or newer and an NVIDIA GPU are recommended.

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # should print ... True
```

Install the CUDA build of PyTorch first; a plain `pip install torch` may give a CPU-only build, which is far too slow for training.

## 2. Data

**Dogs vs. Cats.** The dataset is supplied by the teaching team and is not included in this repository. Place the course archive at `data/datasets.zip`, then run

```bash
python section1/prepare_data.py   # extracts to data/datasets/ and audits all images
python section2/prepare_data.py   # checks the expected layout and image counts
```

The expected layout is `data/datasets/train/{cat,dog}`, `data/datasets/val/{cat,dog}` and `data/datasets/test/`. The six duplicate training images excluded in all experiments are listed in `section1/artifacts/excluded_train.json`.

**CIFAR-10.** Downloaded automatically by torchvision into `data/cifar10/` on the first run (about 170 MB).

## 3. Running the experiments

All commands are run from the repository root. Each queue skips runs that already have a `metrics.json` and stops at the first failure.

```bash
# Comparative experiments E1–E5 (Section 7)
python section1/run_all.py

# Main model, hyperparameter study and multi-seed runs (Sections 4–5)
python section2/run_all.py --data data/datasets --amp --workers 4
python section2/predict.py --run section2/results/MF_seed42 --data data/datasets \
    --output section2/artifacts/submission.csv
python section2/summarize_results.py
python section2/architecture.py

# CIFAR-10 and class imbalance (Sections 8–9)
python section3/run_all.py --data data/cifar10 --amp --workers 4 \
    --queue C1:42 C2:42 H0:42 H1:42 H2:42
python section3/summarize_results.py
```

Use `--workers 2` on machines with fewer CPU cores. `section3/summarize_results.py` summarises whatever runs exist under `section3/results/`; run it only where all CIFAR-10 runs are present, otherwise the CSV files in `section3/artifacts/` will be overwritten with a partial set.

Approximate run times: C1 took about 1.5 minutes and each H run under 1 minute on an NVIDIA L40S; C2 took about 10 minutes on an RTX 3060 Laptop GPU; each Dogs vs. Cats run took 8–30 minutes depending on the GPU.

## 4. Rebuilding the report

The report source is `section1/documents/final_report.md`. With Pandoc and Google Chrome installed:

```bash
cd section1/documents
pandoc final_report.md -s --mathml --css report.css --embed-resources --resource-path=. -o final_report.html
google-chrome --headless --no-pdf-header-footer --print-to-pdf=final_report.pdf final_report.html
```

## Group members

LIU JIALE, ZHANG XUHUA, ZHU ZHENGSHUAI. Individual contributions are listed at the start of the report.
