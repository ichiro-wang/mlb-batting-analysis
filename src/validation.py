"""
File: validation.py
Description: Functions for model validation/cross-validation
"""

from sklearn.model_selection import StratifiedKFold


def create_stratified_kfolds(n_splits: int = 5, random_state: int = 42):
    """
    creating stratified kfolds to be used in cross validation
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
