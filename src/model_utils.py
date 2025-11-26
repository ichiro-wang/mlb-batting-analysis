"""
File: utils.py
Description: Helper functions that don't belong in other files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE


def create_train_test_split(
    X: pd.DataFrame, y: pd.Series, train_size: float = 0.8, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    split the data to a train and test set
    """
    return train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )


def apply_pca(
    data: pd.DataFrame, n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, PCA]:
    """
    apply PCA to reduce dimensions
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(data)

    print(f"PC1 explains {pca.explained_variance_ratio_[0]:.2%} of the data")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]:.2%} of the data")
    print(
        f"Combined, they explain: {np.cumsum(pca.explained_variance_ratio_)[1]:.2%} of all data variation."
    )

    return pca_result, pca


def apply_tsne(
    data: pd.DataFrame, n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, TSNE]:
    """
    apply t-SNE to reduce dimensions
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_result = tsne.fit_transform(data)

    return tsne_result, tsne


def apply_smote(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    generate synthetic samples to deal with class imbalance
    """
    sm = SMOTE(random_state=random_state)

    X_resampled, y_resampled = sm.fit_resample(X, y)

    resampled_counts = y_resampled.value_counts()
    print(f"\nClass distribution of training set after SMOTE:")
    print(resampled_counts)

    return X_resampled, y_resampled
