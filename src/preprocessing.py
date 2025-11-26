"""
File: preprocessing.py
Description: Functions for preprocessing data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def remove_duplicate_columns(
    data: pd.DataFrame, column_pairs: list[str]
) -> pd.DataFrame:
    """
    verify if two columns are duplicates and then remove one of them

    :param data: pandas dataframe
    :param column_pairs: list of pairs of duplicate columns
    """
    data = data.copy()

    for col1, col2 in column_pairs:
        is_duplicate = data[col1].equals(data[col2])
        print(
            f"\nCheck if '{col1}' and '{col2}' columns are duplicates: {is_duplicate}"
        )
        if is_duplicate:
            print(f"'{col2}' removed")
            data = data.drop(columns=[col2])

    return data


def fix_dollar_column(data: pd.DataFrame, column: str = "Dol") -> pd.DataFrame:
    """
    fixing problematic dollar column

    :param data: pandas dataframe
    :param column: column name
    """
    data = data.copy()

    # parentheses represent negative dollar values
    data[column] = (
        data[column]
        .str.replace("$", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .astype(float)
    )
    print(f"\nFixed '{column}' format")

    return data


def rename_column(
    data: pd.DataFrame, original_name: str, new_name: str
) -> pd.DataFrame:
    data = data.copy()
    data[new_name] = data[original_name]
    data.drop(columns=[original_name], inplace=True)
    return data


def convert_sb_cs_rate(data: pd.DataFrame) -> pd.DataFrame:
    """
    instead of counting SB and CS, we convert it to a rate to check efficiency

    SB% = SB / (SB + CS)

    This approach aligns with the project's philosophy of using rate stats instead of counting stats
    """
    data = data.copy()

    data["SB%"] = data["SB"] / (data["SB"] + data["CS"])
    data["SB%"] = data["SB%"].fillna(0)
    data = data.drop(columns=["SB", "CS"])

    return data


def filter_by_threshold(
    data: pd.DataFrame, column: str, threshold: int | float, direction: str = "ge"
) -> pd.DataFrame:
    """
    filter the data by a column threshold

    :param data: pandas dataframe
    :param column: the column you want to filter
    :param threshold: the threshold for keeping/removing values
    :param direction: filter by 'ge' or 'le' (>= or <=). add other methods if needed
    """
    data = data.copy()

    if direction == "ge":
        data = data[data[column] >= threshold]
    else:
        data = data[data[column] <= threshold]
    print(f"\nRemaining samples after filtering by '{column}': {len(data)}")
    return data


def create_performance_tiers(data: pd.DataFrame) -> pd.DataFrame:
    """
    classify players into tiers based on wRC+

    these tiers are not hard set, but rather estimates of hitter level
    """
    data = data.copy()

    def create_tiers(wrc_plus: float) -> str:
        if wrc_plus >= 130:
            return "Elite"
        elif wrc_plus >= 110:
            return "Above Average"
        elif wrc_plus >= 90:
            return "Average"
        else:
            return "Below Average"

    data["Performance Tier"] = data["wRC+"].apply(create_tiers)
    return data


def remove_features(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    remove features that are missing over `threshold * 100%` of their values
    """
    data = data.copy()

    missing = data.isna().sum()
    missing_proportions = missing / len(data)
    to_drop = missing_proportions[missing_proportions > threshold].index.tolist()

    total_missing = missing.sum()
    print(f"Total missing values before removing: {total_missing}")
    print(f"\nDropping columns with over {int(threshold * 100)}% of values missing:")
    print(to_drop)

    data = data.drop(columns=to_drop)
    return data


def impute_values(data: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    imputing values based on `strategy`
    """
    data = data.copy()

    missing = data.isna().sum()
    to_impute = missing[missing > 0].index.tolist()

    total_missing = missing.sum()
    print(f"Total missing values before imputing: {total_missing}")
    print(f"\nImputing columns with missing values using {strategy}.")
    print(to_impute)

    numerical_features = data.select_dtypes(include=[np.number]).columns
    if strategy == "median":
        data[numerical_features] = data[numerical_features].fillna(
            data[numerical_features].median()
        )
    elif strategy == "mean":
        data[numerical_features] = data[numerical_features].fillna(
            data[numerical_features].mean()
        )
    else:
        data[numerical_features] = data[numerical_features].fillna(
            data[numerical_features].mode()[0]
        )

    total_missing = data.isna().sum().sum()
    print(f"Total missing values after imputing: {total_missing}")

    return data


def fix_team_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # a player missing a team means they were on multiple teams
    data["Team"] = data["Team"].fillna("Multiple")
    data["Team"] = data["Team"].replace("- - -", "Multiple")

    return data


def standardize_features(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    standardize numerical features using sklearn standard scaler
    """
    data = data.copy()

    numerical_features = data.select_dtypes(include=[np.number]).columns

    scaler = StandardScaler()

    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data, scaler


def inverse_standardized_features(
    data: pd.DataFrame, scaler: StandardScaler
) -> pd.DataFrame:
    """
    scale back to original data after standardization

    ensure data is in the same dimensions as it was when standardized
    """
    data = scaler.inverse_transform(data)
    return data


def apply_one_hot_encoding(
    data: pd.DataFrame, columns: list[str], prefix: str = None, drop_first: bool = True
) -> pd.DataFrame:
    """
    apply one-hot encoding to data
    """
    return pd.get_dummies(
        data=data, columns=columns, prefix=prefix, drop_first=drop_first
    )
