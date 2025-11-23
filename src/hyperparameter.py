"""
File: outlier_detection.py
Description: Apply hyperparameter tuning on models
"""
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def random_search_rf(rf, X_train, Y_train, X_test, y_test, n_iter, cv, scoring="accuracy", random_state=42):
    """
    Perform RandomizedSearchCV on a RandomForestClassifier and return results
    in the same structure as evaluate_random_forest().
    """

    param_dist = {
        "n_estimators": np.arange(100, 700, 50),
        "max_depth": [None] + list(np.arange(5, 50, 5)),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 20),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    # Run tuning
    random_search.fit(X_train, Y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("\n===== Random Search Results =====")
    print("Best CV Accuracy:", best_score)
    print("Best Parameters:", best_params)

    # Predictions using tuned model
    y_pred_tuned = best_model.predict(X_test)
    y_prob_tuned = best_model.predict_proba(X_test)[:, 1]  # for ROC curve

    # Metrics formatted same as evaluate_random_forest
    tuned_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_tuned),
        "Precision": precision_score(y_test, y_pred_tuned, average='weighted'),
        "Recall": recall_score(y_test, y_pred_tuned, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred_tuned, average='weighted'),
        "CV_mean_accuracy": best_score
    }

    return best_model, tuned_metrics, y_test, y_pred_tuned, y_prob_tuned, X_train, Y_train