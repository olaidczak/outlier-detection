from itertools import product
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import deadwood
from pathlib import Path
import matplotlib.pyplot as plt


def get_predictions(model, X):
    raw = model.fit_predict(X)
    return (raw == -1).astype(int)  # all sklearn outlier models -1 outlier, 1 inlier


def compute_metrics(y_true, y_pred):
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = np.nan
    return {
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
    }


def evaluate(X, y, verbose = True, dataset_name=None, dataset_n=None, dataset_p=None):
    param_grids = {
        "OneClassSVM": {
            "nu": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            "kernel": ["rbf", "poly", "linear"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        },
        "IsolationForest": {
            "contamination": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            "n_estimators": [10, 50, 100, 200, 300],
        },
        "LocalOutlierFactor": {
            "n_neighbors": [5, 10, 20, 30, 50],
            "contamination": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        },
        "DBSCAN": {"eps": np.linspace(0.1, 2.0, 20), "min_samples": [3, 5, 10, 15, 20]},
        "Deadwood": {
            "M": [1, 3, 5, 10, 25, 50],
            "contamination": ["auto", 0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        },
    }

    models = {
        "OneClassSVM": OneClassSVM(cache_size=4096),
        "IsolationForest": IsolationForest(),
        "LocalOutlierFactor": LocalOutlierFactor(),
        "DBSCAN": DBSCAN(),
        "Deadwood": deadwood.Deadwood(),
    }
    results = []
    for model_name, model in models.items():
        param_grid = param_grids[model_name]
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        calculated_count = 0
        total_count = 0
        for combo in product(*values):
            total_count += 1
            params = dict(zip(keys, combo))

            if model_name == "OneClassSVM":
                if params.get("kernel") == "linear" and params.get("gamma") != "scale":
                    continue
            if verbose:
                print(params)
            model.set_params(**params)
            y_pred = get_predictions(model, X)

            if len(np.unique(y_pred)) < 2:
                results.append(
                    {
                        "dataset": dataset_name,
                        "n": dataset_n,
                        "p": dataset_p,
                        "model": model_name,
                        "failure": True,
                    }
                )
                continue

            calculated_count += 1
            metrics = compute_metrics(y, y_pred)
            results.append(
                {
                    "dataset": dataset_name,
                    "n": dataset_n,
                    "p": dataset_p,
                    "model": model_name,
                    **params,
                    **metrics,
                    "failure": False,
                }
            )
        if verbose:
            print(
                f"Evaluated {dataset_name} on {model_name} {calculated_count}/{total_count}"
            )

    return pd.DataFrame(results)


def plot_2dim(dataset_names, data_dir):
    model_classes = {
        "OneClassSVM": OneClassSVM,
        "IsolationForest": IsolationForest,
        "LocalOutlierFactor": LocalOutlierFactor,
        "DBSCAN": DBSCAN,
        "Deadwood": deadwood.Deadwood,
    }
    param_keys = {
        "OneClassSVM": ["nu", "kernel", "gamma"],
        "IsolationForest": ["contamination", "n_estimators"],
        "LocalOutlierFactor": ["n_neighbors", "contamination"],
        "DBSCAN": ["eps", "min_samples"],
        "Deadwood": ["M", "contamination"],
    }
    int_params = {"n_estimators", "n_neighbors", "min_samples"}

    n_cols = len(model_classes) + 1
    fig, axes = plt.subplots(
        len(dataset_names), n_cols, figsize=(3 * n_cols, 3 * len(dataset_names))
    )

    for i, name in enumerate(dataset_names):
        X = np.loadtxt(data_dir / f"{name}.data.gz", ndmin=2)
        y = np.loadtxt(data_dir / f"{name}.labels0.gz", dtype="int")
        y = (y == 0).astype(int)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float64)

        res = evaluate(X_scaled, y, False, name, X.shape[0], X.shape[1])
        p, r = res["Precision"], res["Recall"]
        res["F1"] = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
        res["Score"] = (res["F1"] + res["BalAcc"]) / 2
        # best = (
        #     res[res["failure"] == False]
        #     .sort_values("BalAcc", ascending=False)
        #     .groupby("model")
        #     .first()
        #     .reset_index()
        # )
        best = (
            res[res["failure"] == False]
            .sort_values("Score", ascending=False)
            .groupby("model")
            .first()
            .reset_index()
        )

        axes[i, 0].scatter(X[:, 0], X[:, 1], c=1-y, s=10)
        axes[i, 0].set_title(f"{name}\nTrue labels")

        for j, (model_name, model_cls) in enumerate(model_classes.items()):
            ax = axes[i, j + 1]
            row = best[best["model"] == model_name]
            # if len(row) == 0:
            #     ax.set_title(f"{model_name}\n(failed)")
            #     ax.scatter(X[:, 0], X[:, 1], c="lightgray", s=10)
            #     continue
            row = row.iloc[0]
            params = {}
            for k in param_keys[model_name]:
                v = row[k]
                if k in int_params:
                    v = int(v)
                params[k] = v

            model = model_cls(**params)
            y_pred = (model.fit_predict(X_scaled) == -1).astype(int)

            ax.scatter(X[:, 0], X[:, 1], c=1-y_pred, s=10)
            param_str = ", ".join(f"{k}={params[k]}" for k in param_keys[model_name])
            # ax.set_title(
            #     f"{model_name} (BalAcc={row['BalAcc']:.3f})\n{param_str}", fontsize=8
            # )
            ax.set_title(
                f"{model_name}\n{param_str}\nBalAcc={row['BalAcc']:.3f}, F1 ={row['F1']:.3f}, Score = {row['Score']:.3f}", fontsize=8
            )

    plt.tight_layout()
    plt.show()

def estimate_contamination_iqr(X, k=1.5):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    outliers= np.any((X < lower) | (X > upper), axis=1)

    contamination = outliers.mean()
    return contamination
