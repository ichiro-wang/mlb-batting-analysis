"""
File: validation.py
Description: Functions for model validation/cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def create_stratified_kfolds(n_splits: int = 5, random_state: int = 42):
    """
    creating stratified kfolds to be used in cross validation
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def create_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    label_order: list[str],
) -> np.ndarray:
    """
    create a confusion matrix
    """
    return confusion_matrix(y_test, y_pred, labels=label_order)


def calculate_metrics(
    y_test: pd.Series, y_pred: pd.Series, cross_val_scores: np.ndarray = None
) -> dict[str, any]:
    """
    calculate various metrics
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
    }
    if cross_val_scores is not None:
        metrics["CV mean accuracy"] = np.mean(cross_val_scores)
        metrics["CV std accuracy"] = np.std(cross_val_scores)

    return metrics


def show_metrics(metrics: dict[str, any], title: str) -> None:
    """
    Display classification metrics in a readable format.
    """
    print(f"\n{title}:")
    for metric, value in metrics.items():
        print(f"\t{metric}: {value:.4f}")
