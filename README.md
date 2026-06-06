# Outlier Detection

Comparison of outlier detection algorithms on benchmark datasets, with an ensemble prediction on unlabelled test data.
Notebook contains aggregations and plots needed for analysis, the reasoning and conclusions are in the pdf report.

## Structure

```
├── data/
│   ├── *.mat                – 18 benchmark datasets (ODDS collection) in .mat format
│   └── 2dim/                - folder with 2 dimensional data
│       ├── *.data.gz        – features
│       └── *.labels0.gz     – corresponding labels
├── src/
│   ├── solution.ipynb       – main notebook with four sections:
│   │                          1. grid-search evaluation on all 18 odds datasets
│   │                          2. evaluation and visualisation on 2D datasets
│   │                          3. analysis: rankings, heatmaps, effect of n and p
│   │                          4. EDA and ensemble prediction on unlabelled test data
│   ├── eval.py              – standalone script that runs the full benchmark evaluation
│   ├── utils.py             – helpers:
│   │                            evaluate()                  – grid-search over all models
│   │                            compute_metrics()           – AUC, Acc, Precision, Recall, BalAcc
│   │                            get_predictions()           – normalises sklearn -1/1 output
│   │                            plot_2dim()                 – side-by-side decision boundary plots
│   │                            estimate_contamination_iqr() – IQR-based contamination estimate
│   ├── results_full.csv     – evaluation results across all datasets and models
│   └── test_data.csv
└── requirements.txt
```

## Setup
Python version: 3.11.15

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
