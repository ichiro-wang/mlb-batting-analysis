"""
File: outlier_detection.py
Description: Apply outlier detection algorithms on data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_outliers_isolation_forest(
    X: pd.DataFrame, contamination: float | str = "auto", random_state: int = 42
) -> tuple[IsolationForest, np.ndarray]:
    """
    detect outliers using an isolation forest

    :param X: pandas dataframe
    :param contamination: the proportion of outliers in the data set

    :returns iso_forest: the isolation forest model
    :returns y_pred: list of +1 for inliers and -1 for outliers
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = iso_forest.fit_predict(X)
    return iso_forest, y_pred


def detect_outliers_lof(
    X: pd.DataFrame, contamination: float | str = "auto", n_neighbors: int = 20
) -> tuple[LocalOutlierFactor, np.ndarray, np.ndarray]:
    """
    detect outliers using Local Outlier Factor (LOF)
    """

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, novelty=False
    )
    y_pred = lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_
    return lof, y_pred, scores


def analyze_outliers(outliers_data: pd.DataFrame) -> None:
    """
    provide some analysis on the outliers
    """
    print(f"Number of outliers: {len(outliers_data)}")

    top_n = 10
    print(f"\nTop {top_n} outliers by wRC+:")
    print(outliers_data.nlargest(top_n, "wRC+"))
    print(f"\nBottom {top_n} outliers by wRC+:")
    print(outliers_data.nsmallest(top_n, "wRC+"))
