import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_outliers_isolation_forest(
    X: pd.DataFrame, contamination: float | str = "auto"
) -> tuple[IsolationForest, np.ndarray]:
    """
    detect outliers using an isolation forest

    :param X: pandas dataframe
    :param contamination: the proportion of outliers in the data set

    :returns iso_forest: the isolation forest model
    :returns y_pred: list of +1 for inliers and -1 for outliers
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    y_pred = iso_forest.fit_predict(X)
    return iso_forest, y_pred
