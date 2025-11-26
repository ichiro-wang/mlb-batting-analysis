"""
File: classification.py
Description: Apply classification algorithms on data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Literal
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from src.validation import create_stratified_kfolds

def evaluate_classifier(    
    type:Literal["rf", "xgb"],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    evaluate a classifier. specify type (random forest or xgboost)

    :param type: random forest or xgboost
    :type type: Literal["rf", "xgb"]
    """
    model = None
    if type == "rf":
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    else:
        model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, random_state=random_state)
    
    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)
    
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "CV mean accuracy": np.mean(cross_val_scores),
        "CV std accuracy": np.std(cross_val_scores),
    }
    
    return model, metrics, y_pred


def evaluate_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    **Use evaluate_classifier instead**

    Evaluate a Random Forest classifier.
    """
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    cross_val_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluate  metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "CV mean accuracy": np.mean(cross_val_scores),
        "CV std accuracy": np.std(cross_val_scores),
    }

    return rf, metrics, y_pred


def evaluate_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    **Use evaluate_classifier instead**

    evaluate an XGBoost classifier
    """
    clf = xgb.XGBClassifier(
        tree_method="hist",
        early_stopping_rounds=2,
        random_state=random_state,
    )

    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    cross_val_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

    clf.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred = clf.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "CV mean accuracy": np.mean(cross_val_scores),
        "CV std accuracy": np.std(cross_val_scores),
    }

    return clf, metrics, y_pred


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
