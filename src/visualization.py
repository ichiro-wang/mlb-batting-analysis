"""
File: visualization.py
Description: Helper functions for plotting and visualizing data
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns


def plot_distributions(
    data: pd.DataFrame, key_stats: list[str], type: str = "histograms"
) -> None:
    """
    plotting distrbutions of given features

    :param data: pandas dataframe
    :param key_stats: list of key stats to plot
    :param type: type of plot you want (histograms or box plots)
    """
    cols = 4
    rows = math.ceil(len(key_stats) / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(15, 12))
    axs = axs.ravel()

    for i, stat in enumerate(key_stats):
        if type == "histograms":
            axs[i].hist(data[stat], bins=30, edgecolor="grey")
        elif type == "box plots":
            axs[i].boxplot(data[stat])
        else:
            pass
        axs[i].set_title(f"Distribution of {stat}")
        axs[i].set_xlabel(stat)
        axs[i].set_ylabel("Freq.")

    # hide remaining
    for ax in axs[len(key_stats) :]:
        ax.set_visible(False)

    plt.suptitle(f"{type.title()} of Key Numerical Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_heatmap(data: pd.DataFrame, key_stats: list[str]) -> None:
    corr_matrix = data[key_stats].corr()
    rows, cols = len(corr_matrix), len(corr_matrix.columns)

    high_corr = []
    high_corr_threshold = 0.8
    for row in range(rows):
        for col in range(row + 1, cols):
            corr_val = corr_matrix.iloc[row, col]
            if abs(corr_val) > high_corr_threshold:
                high_corr.append(
                    [
                        corr_val,
                        corr_matrix.columns[row],
                        corr_matrix.columns[col],
                    ]
                )

    print(f"Features with high correlation (>= +-{high_corr_threshold}):")
    high_corr.sort(reverse=True)
    for corr, feat1, feat2 in high_corr:
        print(f"{feat1}, {feat2}: {corr:.4f}")

    plt.figure(figsize=(12, 10))
    plt.title("Correlation Heatmap of Key Statistics", pad=10)
    sns.heatmap(corr_matrix, cmap="vlag", annot=True, fmt=".2f", center=0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_tier_distribution(data: pd.DataFrame, tier_order: list[str]) -> None:
    """
    after adding performance tiers to data, plot its distribution
    """
    print(f"\n{data["Performance Tier"].value_counts()}")

    plt.figure(figsize=(10, 6))
    data["Performance Tier"].value_counts().reindex(tier_order).plot(kind="bar")
    plt.title("Performance Tier Distribution", pad=10)
    plt.ylabel("Count")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_pca(
    pca_result: np.ndarray,
    labels: pd.Series | np.ndarray | None = None,
    label_order: list[str] | None = None,
    title: str = "PCA Visualization of Batter Data",
    colors: dict[float, str] | None = None,
    contamination: float | None = None,
    alpha: float | dict[str, float] = 0.7,
) -> None:
    """
    use scatter plot to visualize PCA
    """
    plt.figure(figsize=(10, 6))

    if labels is None:
        plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            alpha=alpha if isinstance(alpha, float) else 0.7,
            edgecolors="k",
            color="red",
        )
    else:
        if not colors:
            palette = sns.color_palette("deep", len(label_order))
            colors = dict(zip(label_order, palette))

        for label in label_order:
            mask = labels == label
            plt.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                label=label,
                alpha=alpha[label] if isinstance(alpha, dict) else alpha,
                edgecolors="k",
                color=colors[label],
            )
        plt.legend(loc="upper left")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if contamination is not None:
        title += f" (contamination={contamination})"
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    
def plot_best_k(data,method, best_k, random_state = 42):
    """
    Runs KMeans with the best k, applies PCA for 2D visualization,
    and plots the clusters.
    """
    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)
    PCA_model = PCA(n_components=2)
    pca_result = PCA_model.fit_transform(data)
    plt.figure(figsize=(7, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, s=40)
    plt.title(f"K-Means Clustering (k={best_k}) using {method} (PCA Reduced)")
    plt.xlabel("PCA Component 1");
    plt.ylabel("PCA Component 2");
    plt.grid(True)
    plt.show()
