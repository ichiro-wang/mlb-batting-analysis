"""
File: feature_selection.py
Description: Apply feature selection algorithms on data
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def remove_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    removing features based on domain knowledge
    """
    data = data.copy()

    plus_stats = [col for col in data.columns if col.endswith("+")]
    base_names = {col.rstrip("+") for col in plus_stats}
    non_plus_stats = [base for base in base_names if base in data.columns]

    pitch_types = ["SL", "CT", "CB", "CH", "SF"]
    other_pitch_stats = ["FB% (Pitch)", "FBv", "wFB", "wFB/C", "XX%"]
    for type in pitch_types:
        other_pitch_stats.extend([f"{type}%", f"{type}v", f"w{type}", f"w{type}/C"])

    to_remove = {
        "non_plus_stats": non_plus_stats,
        "pitch_specific_stats": [
            col for col in data.columns if "(sc)" in col or "(pi)" in col
        ],
        "other_pitch_stats": other_pitch_stats,
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
            "R",
            "RBI",
            "IBB",
            "HBP",
            "SF",
            "SH",
            "GDP",
            "HardHit",
            "Barrels",
        ],
        "value_stats": [
            "WAR",
            "L-WAR",
            "RAR",
            "Dol",
            "Bat",
            "Fld",
            "Pos",
            "Rep",
            "wRAA",
            "BsR",
            "Off",
            "Def",
            "Lg",
        ],
        "context_stats": [
            "WPA",
            "-WPA",
            "+WPA",
            "RE24",
            "REW",
            "pLI",
            "phLI",
            "PH",
            "WPA/LI",
            "Clutch",
        ],
        "other_stats": ["OPS", "wOBA", "xwOBA", "ISO+", "BABIP+"],
    }

    for name, stats in to_remove.items():
        data.drop(columns=stats, inplace=True, errors="ignore")
        print(f"Removing {name}: {stats}")

    return data


def apply_rfecv(X_train: pd.DataFrame, y_train: pd.Series) -> RFECV:
    """
    apply recursive feature elimination with cross validation to find best features
    """
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    # use a stratified n_splits strategy
    n_splits = 5
    cv_split = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    step_size = 1

    rfecv = RFECV(
        estimator=rf,
        step=step_size,  # adjust step to increase/decrease speed
        cv=cv_split,
        scoring="accuracy",
        n_jobs=-1,
    )

    rfecv.fit(X=X_train, y=y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Optimal features: {rfecv.get_feature_names_out()}")

    return rfecv

def apply_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    Cs: np.ndarray = None,
    cv: int = 5,
    tol: float = 1e-4,
) -> tuple[Pipeline, pd.Index]:
    """
    L1-penalized Logistic Regression (LASSO-like) for classification feature selection.
    """

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logregcv",
                LogisticRegressionCV(
                    Cs=Cs,
                    cv=cv,
                    penalty="l1",
                    solver="saga",
                    multi_class="ovr",
                    scoring="accuracy",
                    max_iter=5000,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    model = pipe.named_steps["logregcv"]

    coef_matrix = model.coef_
    nonzero_mask = (np.abs(coef_matrix) > tol).any(axis=0)

    selected_features = X_train.columns[nonzero_mask]
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:", selected_features.tolist())

    return pipe, selected_features

