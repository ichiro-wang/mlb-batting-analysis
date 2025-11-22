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
from sklearn.metrics import confusion_matrix, roc_curve, auc
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


def plot_pca_by_tier(
    pca_result: np.ndarray, tiers: pd.Series, tier_order: list[str]
) -> None:
    """
    use scatter plot to visualize PCA, coloured by performance tiers
    """
    palette = sns.color_palette("deep", len(tier_order))
    colors = dict(zip(tier_order, palette))

    plt.figure(figsize=(10, 6))
    # PCA coloured by tiers
    for tier in tier_order:
        mask = tiers == tier
        plt.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            label=tier,
            alpha=0.7,
            edgecolors="k",
            color=colors[tier],
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization of Batter Data")
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_outliers(
    pca_result: np.ndarray, y_pred: np.ndarray, contamination: float = None
):
    """
    plotting outliers on a scatter plot
    """
    labels = {"inlier": 1, "outlier": -1}
    colors = {"inlier": "blue", "outlier": "red"}

    plt.figure(figsize=(10, 6))
    for label, value in labels.items():
        mask = y_pred == value
        alpha = 1 if label == "outlier" else 0.6
        plt.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            label=label,
            alpha=alpha,
            edgecolors="k",
            color=colors[label],
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    title = "PCA Visualization of Outliers"
    if contamination is not None:
        title += f" (contamination={contamination})"
    plt.title(title)
    plt.legend()
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
    
def plot_confusion_matrix(y_test, y_pred, title):
    """
    Plots the confusion matrix for classification results.
    """

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def show_classfication_metrics(metrics: dict, model) -> None:
    """
    Display classification metrics in a readable format.
    """
    print(f"\n{model} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
