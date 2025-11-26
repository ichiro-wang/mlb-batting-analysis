"""
File: hyperparameter.py
Description: Apply hyperparameter tuning on models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.validation import create_stratified_kfolds


def random_search_rf(
    rf: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_iter: int = 50,
    n_splits: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
):
    """
    Perform RandomizedSearchCV on a RandomForestClassifier and return results
    in the same structure as evaluate_random_forest().
    """

    param_dists = {
        "n_estimators": np.arange(100, 700, 50),
        "max_depth": [None] + list(np.arange(5, 50, 5)),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 20),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dists,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=1,
        random_state=random_state,
        n_jobs=1,  # n_jobs = -1 in the random forest. don't do it here
    )

    # Run tuning
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("\n===== Random Search Results =====")
    print("Best CV Accuracy:", best_score)
    print("Best Parameters:", best_params)

    # Predictions using tuned model
    y_pred_tuned = best_model.predict(X_test)

    # Metrics formatted same as evaluate_random_forest
    tuned_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_tuned),
        "Precision": precision_score(y_test, y_pred_tuned, average="weighted"),
        "Recall": recall_score(y_test, y_pred_tuned, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred_tuned, average="weighted"),
        "CV_mean_accuracy": best_score,
    }

    return best_model, tuned_metrics, y_pred_tuned


def random_search_xgb(
    xgb: xgb.XGBModel,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_iter: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
):
    pass
