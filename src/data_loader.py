import pandas as pd


def load_cleaned_data(
    filepath: str = "../data/processed/batting_2015_2024_cleaned.csv",
):
    return pd.read_csv(filepath)
