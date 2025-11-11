import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Literal


class OutlierDetector:
    """
    a class representing an outlier detection model

    Attributes:
        method: the outlier detector method we want to use. 'iso_forest' or 'lof' (we can choose other ones too)
        contamination: the proportion of outliers in our data set. default 'auto'
        model: the actual outlier detector model
    """

    def __init__(
        self,
        method: Literal["iso_forest", "lof"] = "iso_forest",
        contamination: float | str = "auto",
    ) -> None:
        """
        :param method: outlier detection method
        :param contamination: percentage of outliers
        """
        self.method = method
        self.contamination = contamination

        random_state = 42
        if method == "iso_forest":
            self.model = IsolationForest(
                random_state=random_state, n_jobs=-1, contamination=contamination
            )
        else:
            pass

    def find_outliers(self, X: pd.DataFrame) -> np.ndarray:
        """
        :param X: the data set
        :returns y_pred: array where 1 is an inlier and -1 is an outlier
        """
        if self.method == "iso_forest":
            y_pred = self.model.fit_predict(X)

        return y_pred

    def analyze_outliers(
        self, X: pd.DataFrame, reference_data: pd.DataFrame
    ) -> dict[str, any]:
        """
        :param X: the data set
        :param reference_data: a reference data set with Names, Seasons, and wRC+

        :returns results: dictionary of various attributes
        """
        y_pred = self.find_outliers(X)
        outliers_mask = y_pred == -1
        num_outliers = outliers_mask.sum()
        rel_freq_outliers = num_outliers / len(X)

        outlier_data = reference_data[outliers_mask]

        results = {
            "y_pred": y_pred,
            "outliers_mask": outliers_mask,
            "num_outliers": num_outliers,
            "rel_freq_outliers": rel_freq_outliers,
            "outlier_data": outlier_data,
        }
        
        return results

