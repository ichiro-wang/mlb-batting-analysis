"""
File: data_loader.py
Description: Loads CSV files from data folder
"""

import pandas as pd


def load_raw_data() -> pd.DataFrame:
    filepath = "../data/raw/batting_2015_2024.csv"
    print(f"Loading raw data from '{filepath}'")
    return pd.read_csv(filepath)


def load_cleaned_data(reference_data: bool = False) -> pd.DataFrame:
    if reference_data:
        filepath = "../data/processed/batting_2015_2024_reference.csv"
    else:
        filepath = "../data/processed/batting_2015_2024_cleaned.csv"
    print(
        f"Loading {'reference' if reference_data else 'cleaned'} data from '{filepath}'"
    )
    return pd.read_csv(filepath)
