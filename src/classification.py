"""
File: classification.py
Description: Apply classification algorithms on data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def evaluate_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
):
    """
    Evaluate a Random Forest classifier.
    Assumes X_train, X_test, y_train, y_test are already prepared in analysis.ipynb.
    """
    rf = RandomForestClassifier(random_state=random_state)
    cross_val_scores = cross_val_score(
        rf, X_train, y_train, cv=cv_folds, scoring="accuracy"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Evaluate  metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "CV_mean_accuracy": np.mean(cross_val_scores),
        "CV_std_accuracy": np.std(cross_val_scores),
    }
    return rf, metrics, y_test, y_pred, X_train, y_train


def evaluate_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
):
    pass


def show_classfication_metrics(metrics: dict[str, any], model: str) -> None:
    """
    Display classification metrics in a readable format.
    """
    print(f"\n{model} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def create_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> np.ndarray:
    """
    create a confusion matrix
    """
    return confusion_matrix(y_test, y_pred)
