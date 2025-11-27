"""
File: classification.py
Description: Apply classification algorithms on data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Literal
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.shared import (
    create_stratified_kfolds,
    create_confusion_matrix,
    calculate_metrics,
    show_metrics,
)


def evaluate_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_splits: Literal[5, 10] = 5,
    random_state: int = 42,
):
    """
    Evaluate a Random Forest classifier using 5 or 10 fold cross-validation
    """
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    cross_val_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluate  metrics
    metrics = calculate_metrics(
        y_test=y_test, y_pred=y_pred, cross_val_scores=cross_val_scores
    )

    return rf, metrics, y_pred


def evaluate_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_splits: Literal[5, 10] = 5,
    num_boost_round: int = 100,
    random_state: int = 42,
):
    """
    Evaluate an XGBoost classifier using 5 or 10 fold cross-validation
    """
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    dtrain = xgb.DMatrix(data=X_train, label=y_train_encoded)

    params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "num_class": len(y_train.unique()),
        "random_state": random_state,
    }

    folds = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    # cross validation
    cv_res = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        folds=folds,
        metrics=["mlogloss", "merror"],
        seed=random_state,
        callbacks=[xgb.callback.EarlyStopping(10)],
    )

    # find the best iteration after cross-val: the one with least test logloss
    best_iteration = cv_res["test-mlogloss-mean"].argmin()

    model = xgb.XGBClassifier(
        n_estimators=best_iteration + 1,
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(y_train.unique()),
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)

    metrics = calculate_metrics(y_test=y_test, y_pred=y_pred)

    return model, metrics, y_pred


def train_and_evaluate_models(
    classifier_type: Literal["rf", "xgb"],
    datasets: dict[str, dict[str, pd.DataFrame]],
    label_order: list[str],
) -> dict[str, dict[str, any]]:
    res = {}

    for name, dataset in datasets.items():
        X_train, X_test = dataset["X_train"], dataset["X_test"]
        y_train, y_test = dataset["y_train"], dataset["y_test"]

        if classifier_type == "rf":
            model, metrics, y_pred = evaluate_random_forest(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )
            title = f"Random Forest metrics ({name})"

        elif classifier_type == "xgb":
            model, metrics, y_pred = evaluate_xgboost(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )
            title = f"XGBoost metrics ({name})"

        else:
            raise ValueError(f"Unsupported classifier: {classifier_type}")

        show_metrics(metrics=metrics, title=title)

        cm = create_confusion_matrix(
            y_test=y_test, y_pred=y_pred, label_order=label_order
        )

        res[name] = {"model": model, "cm": cm, "y_pred": y_pred}

    return res
