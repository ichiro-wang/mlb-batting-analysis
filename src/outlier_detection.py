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

        self.y_pred = None # array where 1 is an inlier and -1 is an outlier
        self.outliers_mask = None # array where True if outlier and False if inlier
        self.num_outliers = None
        self.rel_freq_outliers = None

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
        self.y_pred = self.find_outliers(X)
        self.outliers_mask = self.y_pred == -1
        self.num_outliers = self.outliers_mask.sum()
        self.rel_freq_outliers = self.num_outliers / len(X)

        self.outlier_data = reference_data[self.outliers_mask]

        results = {
            "y_pred": self.y_pred,
            "outliers_mask": self.outliers_mask,
            "num_outliers": self.num_outliers,
            "rel_freq_outliers": self.rel_freq_outliers,
            "outlier_data": self.outlier_data,
        }

        return results

    def outlier_summary(self, outlier_data: pd.DataFrame | None = None, n=10) -> None:
        """
        :param outlier_data: outlier data found during `analyze_outliers`
        :returns None:
        """
        print("Summary of outliers")
        print(f"Total number of samples: {len(self.y_pred)}")
        print(
            f"Number of outliers: {self.num_outliers}, Proportion of outliers: {self.rel_freq_outliers}"
        )

        if outlier_data is None:
            outlier_data = self.outlier_data

        top_outliers = outlier_data.nlargest(n=n, columns=["wRC+"])
        bottom_outliers = outlier_data.nsmallest(n=n, columns=["wRC+"])
        print(f"\nTop outliers:\n{top_outliers}")
        print(f"\nBottom outliers:\n{bottom_outliers}")
