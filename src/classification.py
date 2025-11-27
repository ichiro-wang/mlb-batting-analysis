"""
File: classification.py
Description: Apply classification algorithms on data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Literal
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.shared import create_stratified_kfolds, calculate_metrics


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
    y_test_encoded = le.transform(y_test)

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

    # model = xgb.train(params=params, dtrain=dtrain, num_boost_round=best_iteration + 1)

    model = xgb.XGBClassifier(
        n_estimators=best_iteration + 1,
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(y_train.unique()),
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)

    metrics = calculate_metrics(y_test=y_test, y_pred=y_pred)

    return model, metrics, y_pred


def show_classfication_metrics(metrics: dict[str, any], title: str) -> None:
    """
    Display classification metrics in a readable format.
    """
    print(f"\n{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def create_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    label_order: list[str],
) -> np.ndarray:
    """
    create a confusion matrix
    """
    return confusion_matrix(y_test, y_pred, labels=label_order)
