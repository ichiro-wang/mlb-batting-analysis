"""
File: hyperparameter.py
Description: Apply hyperparameter tuning on models
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.shared import create_stratified_kfolds, calculate_metrics


def random_search(
    classifier: RandomForestClassifier | XGBClassifier,
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
    Perform RandomizedSearchCV on a classifier and return results
    in the same structure as evaluate_classifier() in `classification.py`.
    """
    if isinstance(classifier, RandomForestClassifier):
        param_dists = {
            "n_estimators": np.arange(200, 600, 50),
            "max_depth": np.concatenate([
                np.arange(5, 16, 5), np.arange(20, 51, 10), np.array([None])
            ]),
            "min_samples_split": np.arange(5, 21, 5),
            "min_samples_leaf": np.concatenate([np.array([1]), np.arange(2, 9, 2)]),
            "max_features": np.array(["sqrt", "log2", None]),
        }

        model_name = "Random Forest"

    elif isinstance(classifier, XGBClassifier):
        param_dists = {
            "learning_rate": np.arange(0.05, 0.31, 0.05),
            "max_depth": np.concatenate([
                np.arange(2, 7), np.arange(8, 13, 2), np.array([15])
            ]),
            "min_child_weight": np.arange(1, 8, 2),
            "gamma": np.arange(0, 0.5, 0.1),
            "colsample_bytree": np.concatenate([np.arange(0.3, 0.6), np.array([0.7])]),
        }

        model_name = "XGBoost"
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

    else:
        raise ValueError(f"Unsupported classifier: {type(classifier)}")

    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_dists,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=1,
        random_state=random_state,
        n_jobs=1,  # n_jobs = -1 in the classifier. otherwise we get thread contention
    )

    # Run tuning
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f"\nRandom Search Results on {model_name}")
    print("Best CV Accuracy:", best_score)
    print("Best Parameters:")
    for name, val in best_params.items():
        print(f"{name}: {val}")

    # Predictions using tuned model
    y_pred_tuned = best_model.predict(X_test)
    
    if isinstance(classifier, XGBClassifier):
        y_pred_tuned = le.inverse_transform(y_pred_tuned)

    # Metrics formatted same as evaluate_random_forest
    tuned_metrics = calculate_metrics(y_test=y_test, y_pred=y_pred_tuned)

    return best_model, tuned_metrics, y_pred_tuned
