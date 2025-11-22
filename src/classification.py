"""
File: classification.py
Description: Apply classification algorithms on data
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def evaluate_random_forest(X_train, X_test, y_train, y_test, cv_folds=5, random_state=42):
    """
    Evaluate a Random Forest classifier.
    Assumes X_train, X_test, y_train, y_test are already prepared in analysis.ipynb.
    """
    rf = RandomForestClassifier(random_state=random_state)
    cross_val_scores = cross_val_score(rf, X_train, y_train, cv=cv_folds, scoring='accuracy')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    #Evaluate  metrics
    metrics ={
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "CV_mean_accuracy": np.mean(cross_val_scores),
        "CV_std_accuracy": np.std(cross_val_scores)
    }
    if len(np.unique(y_test)) == 2:  # Binary classification
        y_prob = rf.predict_proba(X_test)[:, 1]
        metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
    return metrics, y_test, y_pred, y_prob
    
    
