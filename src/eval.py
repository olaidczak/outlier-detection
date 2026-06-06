import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io
from sklearn.preprocessing import RobustScaler
from utils import evaluate

data_dir = Path("../data")

dataset_names = [
    "annthyroid",
    "arrhythmia",
    "breastw",
    "cardio",
    "glass",
    "letter",
    "lympho",
    "mammography",
    "musk",
    "pendigits",
    "satellite",
    "satimage-2",
    "shuttle",
    "speech",
    "thyroid",
    "vertebral",
    "vowels",
    "wine",
]
scaler = RobustScaler()
all_results = []
for dataset in dataset_names:
    path = data_dir / f"{dataset}.mat"
    mat = scipy.io.loadmat(path)
    X = mat.get("X")
    y = mat.get("y")
    X = pd.DataFrame(X)
    y = pd.Series(y.flatten(), name="y")
    data = pd.concat([X, y], axis=1)
    data.drop_duplicates(inplace=True)
    X = data.drop(columns=["y"])
    y = data["y"]
    n = X.shape[0]
    p = X.shape[1]
    X = scaler.fit_transform(X)
    res = evaluate(X, y, dataset, n, p)
    all_results.append(res)

res = pd.concat(all_results, ignore_index=True)
res.to_csv("results_part3.csv", index=False)
