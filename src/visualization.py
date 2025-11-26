"""
File: visualization.py
Description: Helper functions for plotting and visualizing data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

rect = [0, 0, 1, 0.98]
plot_size = (10, 6)


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
    rows = int(np.ceil(len(key_stats) / cols))

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
    plt.tight_layout(rect=rect)
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
    sns.heatmap(data=corr_matrix, cmap="vlag", annot=True, fmt=".2f", center=0)
    plt.tight_layout(rect=rect)
    plt.show()


def plot_tier_distribution(data: pd.DataFrame, tier_order: list[str]) -> None:
    """
    after adding performance tiers to data, plot its distribution
    """
    print(f"\n{data["Performance Tier"].value_counts()}")

    plt.figure(figsize=plot_size)
    data["Performance Tier"].value_counts().reindex(tier_order).plot(kind="bar")
    plt.title("Performance Tier Distribution", pad=10)
    plt.ylabel("Count")
    plt.tight_layout(rect=rect)
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
    plt.figure(figsize=plot_size)

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
    plt.tight_layout(rect=rect)
    plt.show()


def plot_tsne(
    tsne_result: np.ndarray,
    labels: pd.Series | np.ndarray | None = None,
    label_order: list[str] | None = None,
    title: str = "t-SNE Visualization of Batter Data",
    colors: dict[float, str] | None = None,
    alpha: float | dict[str, float] = 0.7,
) -> None:
    """
    use scatter plot to visualize t-sne
    """
    plt.figure(figsize=plot_size)

    if labels is None:
        plt.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            alpha=alpha if isinstance(alpha, float) else 0.7,
            edgecolors="k",
            color="red",
        )
    else:
        if label_order is None:
            label_order = sorted(np.unique(labels))
        if not colors:
            palette = sns.color_palette("deep", len(label_order))
            colors = dict(zip(label_order, palette))

        for label in label_order:
            mask = labels == label
            plt.scatter(
                tsne_result[mask, 0],
                tsne_result[mask, 1],
                label=label,
                alpha=alpha[label] if isinstance(alpha, dict) else alpha,
                edgecolors="k",
                color=colors[label],
            )
        plt.legend(loc="upper left")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.tight_layout(rect=rect)
    plt.show()


def plot_scores(
    k_list: list[int],
    results: dict[str, any],
) -> None:
    """
    plotting silhouette scores of different clustering methods and different feature selection methods

    subplots for visual comparison
    """
    include_inertia = "inertias" in results["None"]

    cols = 2
    rows = int(np.ceil(len(results) / cols) * (2 if include_inertia else 1))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.ravel()

    feat_select_method_names = list(results.keys())

    for i, method_name in enumerate(feat_select_method_names):
        scores = results[method_name]["silhouette_scores"]

        axs[i].plot(k_list, scores, marker="o")
        axs[i].set_xlabel("Number of Clusters (k)")
        axs[i].set_ylabel("Silhouette Score")
        axs[i].set_title(f"Silhouette Score vs k ({method_name})")

    if include_inertia:
        for i, method_name in enumerate(
            feat_select_method_names, len(feat_select_method_names)
        ):
            scores = results[method_name]["inertias"]

            axs[i].plot(k_list, scores, marker="o")
            axs[i].set_xlabel("Number of Clusters (k)")
            axs[i].set_ylabel("Inertia")
            axs[i].set_title(f"Inertia vs k ({method_name})")

    # hide remaining
    for ax in axs[len(results) * (2 if include_inertia else 1) :]:
        ax.set_visible(False)

    plt.suptitle(f"Scores of Various Feature Selection Methods", fontsize=16)
    plt.tight_layout(rect=rect)
    plt.show()


def plot_clusters(results: dict[str, any], cluster_method: str) -> None:
    """
    plotting clusterings of different clustering methods and feature selection methods

    subplots for visual comparison
    """

    cols = 2
    rows = int(np.ceil(len(results) / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.ravel()

    feat_select_methods = list(results.keys())

    for i, method_name in enumerate(feat_select_methods):
        pca_result = results[method_name]["pca_result"]
        labels = results[method_name]["labels"]

        scatter = axs[i].scatter(
            pca_result[:, 0], pca_result[:, 1], c=labels, cmap="tab10", alpha=0.7
        )
        axs[i].set_title(f"{cluster_method} Clustering (method={method_name})")
        axs[i].set_xlabel("PC1")
        axs[i].set_ylabel("PC1")

        axs[i].legend(*scatter.legend_elements(), title="Clusters", loc="upper right")

    for ax in axs[len(results) :]:
        ax.set_visible(False)

    plt.suptitle(f"{cluster_method} Clusterings", fontsize=16)
    plt.tight_layout(rect=rect)
    plt.show()


def plot_dendrograms(results: dict[str, any], **kwargs) -> None:
    """
    plot dendrograms from hierarchical clustering

    help from: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """
    cols = 2
    rows = int(np.ceil(len(results) / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = axs.ravel()

    feat_select_methods = list(results.keys())

    for i, method_name in enumerate(feat_select_methods):
        model = results[method_name]["model"]
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for j, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[j] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        dendrogram(linkage_matrix, ax=axs[i], **kwargs)
        axs[i].set_title(f"Hierarchical Clustering Dendrogram ({method_name})")
        axs[i].tick_params(axis="x", rotation=60, labelsize=10)

    for ax in axs[len(results) :]:
        ax.set_visible(False)

    plt.suptitle("Hierarchical Clusterings", fontsize=16)
    plt.tight_layout(rect=rect)
    plt.show()


def plot_cluster_radars(
    data: pd.DataFrame, labels: np.ndarray, stats: list[str] = ["AVG+", "OBP+", "SLG+"]
) -> None:
    """
    plotting stats within clusters to spot potential patterns
    """
    cluster_summary = data.groupby(labels)[stats].mean()

    global_min = cluster_summary.min().min()
    global_max = cluster_summary.max().max()

    num_stats = len(stats)
    clusters = cluster_summary.index.tolist()
    num_clusters = len(clusters)

    # angular positions
    angles = np.linspace(0, 2 * np.pi, num_stats, endpoint=False).tolist()
    angles += angles[:1]

    cols = 2
    rows = int(np.ceil(num_clusters / cols))

    fig, axs = plt.subplots(
        rows, cols, figsize=(5 * cols, 5 * rows), subplot_kw=dict(polar=True)
    )
    axs = axs.ravel()

    color_cycle = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for i, cluster in enumerate(clusters):
        ax = axs[i]

        values = cluster_summary.loc[cluster].tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, color=color_cycle[i])
        ax.fill(angles, values, alpha=0.2, color=color_cycle[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(stats)
        ax.set_ylim(global_min, global_max)
        ax.set_title(f"Cluster {cluster}")

    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.suptitle("Key Stats Within Clusters", fontsize=16)
    plt.tight_layout(rect=rect)
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    tier_names: list[str] = ["Below Average", "Average", "Above Average", "Elite"],
) -> None:
    """
    Plots the confusion matrix for classification results.

    :param cm: confusion matrix
    """
    plt.figure(figsize=plot_size)
    plt.title(title)
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=tier_names,
        yticklabels=tier_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout(rect=rect)
    plt.show()
    
def show_classfication_metrics(metrics: dict, model) -> None:
    """
    Display classification metrics in a readable format.
    """
    print(f"\n{model} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
