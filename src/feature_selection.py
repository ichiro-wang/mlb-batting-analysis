import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def remove_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    removing features based on domain knowledge
    """
    data = data.copy()

    to_remove = {
        "counting_stats": [
            "G",
            "AB",
            "PA",
            "H",
            "1B",
            "2B",
            "3B",
            "HR",
            "Pitches",
            "Balls",
            "Strikes",
            "SO",
            "BB",
            "GB",
            "FB",
            "Events",
        ],
        "value_stats": ["WAR", "L-WAR", "RAR", "Dol", "Bat", "Fld", "Pos", "Rep", "wRAA", "BsR", "Off", "Def"],
        "context_stats": ["WPA", "-WPA", "+WPA", "RE24", "REW"],
        "other_stats": ["OPS"]
    }

    for name, stats in to_remove.items():
        data.drop(columns=stats, inplace=True)
        print(f"Removing {name}: {stats}")

    return data


def apply_rfecv(X_train: pd.DataFrame, y_train: pd.Series) -> RFECV:
    """
    apply recursive feature elimination with cross validation to find best features
    """

    # use a stratified n_splits strategy
    n_splits = 5
    cv_split = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)

    rfecv = RFECV(
        estimator=rf,
        step=5,  # adjust step to increase/decrease speed
        cv=cv_split,
        scoring="accuracy",
        n_jobs=-1,
    )

    rfecv.fit(X=X_train, y=y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Optimal features: {rfecv.get_feature_names_out()}")

    return rfecv
