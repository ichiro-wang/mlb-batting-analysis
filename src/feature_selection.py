"""
File: feature_selection.py
Description: Apply feature selection algorithms on data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegressionCV
from src.validation import create_stratified_kfolds


def remove_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    removing features based on domain knowledge
    """
    data = data.copy()

    plus_stats = [col for col in data.columns if col.endswith("+")]
    base_names = {col.rstrip("+") for col in plus_stats}
    non_plus_stats = [base for base in base_names if base in data.columns]

    pitch_types = ["SL", "CT", "CB", "CH", "SF"]
    other_pitch_stats = ["FB% (Pitch)", "FBv", "wFB", "wFB/C", "XX%", "Pace"]
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
            "IFFB",
            "IFH",
            "LD",
            "BUH",
            "BU",
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
            "wSB",
            "UBR",
            "wGDP",
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
        print(f"Removing {len(stats)} {name}: {stats}")

    return data


def apply_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    step_size: int = 1,
    random_state: int = 42,
) -> tuple[RFECV, np.ndarray[str]]:
    """
    apply recursive feature elimination with cross validation to find best features
    """

    # use a stratified n_splits strategy
    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)

    rfecv = RFECV(
        estimator=rf,
        step=step_size,  # adjust step to increase/decrease speed
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    rfecv.fit(X=X_train, y=y_train)
    selected_features = rfecv.get_feature_names_out()

    print(f"Number of selected features: {rfecv.n_features_}")
    print(f"Selected features: {selected_features}")

    return rfecv, selected_features


def apply_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    Cs: np.ndarray | None = None,
    n_splits: int = 5,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[LogisticRegressionCV, np.ndarray[str]]:
    """
    L1-penalized Logistic Regression (LASSO-like) for classification feature selection.
    """

    cv = create_stratified_kfolds(n_splits=n_splits, random_state=random_state)

    lrcv = LogisticRegressionCV(
        Cs=Cs,
        cv=cv,
        penalty="l1",
        solver="saga",
        multi_class="ovr",
        scoring="accuracy",
        max_iter=5000,
        random_state=random_state,
        n_jobs=-1,
    )

    lrcv.fit(X_train, y_train)

    coef_matrix = lrcv.coef_
    nonzero_mask = (np.abs(coef_matrix) > tol).any(axis=0)

    selected_features = X_train.columns[nonzero_mask].to_numpy()
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Selected features: {selected_features.tolist()}")

    return lrcv, selected_features
