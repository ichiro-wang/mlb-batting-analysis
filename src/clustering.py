"""
File: clustering.py
Description: Apply clustering algorithms on data
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def apply_kmeans(
    data: pd.DataFrame,
    k_list: list[int] = list(range(2, 11)),
) -> tuple[int, list[float], list[float], np.ndarray]:
    """
    Runs KMeans for k in range [k_min, k_max], and plots silhouette scores,
    and returns the best k.
    """

    silhouette_scores = []
    inertias = []
    best_labels = []
    best_score = float("-inf")
    best_k = 0

    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)

        if best_score < score:
            best_score = score
            best_k = k
            best_labels = cluster_labels

    return best_k, silhouette_scores, inertias, best_labels


def apply_hdbscan(
    data: pd.DataFrame, min_cluster_size: int = 10, min_samples: int = 5
) -> tuple[HDBSCAN, np.ndarray]:
    """
    apply **HDBSCAN**
    """
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1)
    y_pred = hdb.fit_predict(data)
    return hdb, y_pred


def apply_hierarchical(data: pd.DataFrame, k_list: list[int] = list(range(2, 11))):
    """
    apply **hierarchical** clustering and check silhouette scores of multiple k's
    """
    silhouette_scores = []
    best_labels = []
    best_score = float("-inf")
    best_k = 0
    best_agg = None

    for k in k_list:
        agg = AgglomerativeClustering(n_clusters=k, compute_distances=True)
        labels = agg.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

        if best_score < score:
            best_score = score
            best_k = k
            best_labels = labels
            best_agg = agg

    return best_agg, silhouette_scores, best_k, best_labels


def evaluate_clustering(X: pd.DataFrame, labels: np.ndarray) -> None:
    """
    evaluating clusterings using silhouette, calinski, and davies scores
    """
    # make sure we have enough clusters
    # -1 means noise, so don't count it as a cluster
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.count_nonzero(labels == -1)
    if n_clusters <= 1:
        print(f"Insufficient clusters: {n_clusters}.\nCannot calculate metrics.")
        return

    valid_mask = labels != -1
    if valid_mask.sum() <= 1:
        print(f"Not enough non-noisy points.\nCannot calculate metrics.")
        return

    X_filtered = X[valid_mask]
    labels_filtered = labels[valid_mask]

    silhouette = silhouette_score(X_filtered, labels_filtered)
    calinski = calinski_harabasz_score(X_filtered, labels_filtered)
    davies = davies_bouldin_score(X_filtered, labels_filtered)

    print(f"Clusters: {n_clusters}")
    print(f"Noisy points: {n_noise}")
    print(f"silhouette: {silhouette:.4f}")
    print(f"calinski: {calinski:.4f}")
    print(f"davies: {davies:.4f}")
