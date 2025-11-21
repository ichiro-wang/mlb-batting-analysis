"""
File: clustering.py
Description: Apply clustering algorithms on data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(data, method, k_min = 2, k_max = 10, random_state = 42):
    """
    Runs KMeans for k in [k_min, k_max], plots silhouette scores,
    and returns the best k.
    """

    silhouette_scores = []
    k_list = list(range(k_min, k_max + 1))
    
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(k_list, silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(str(method)+"Feature's Silhouette Scores for Different k")
    plt.grid(True)
    plt.show()
    best_k = k_list[np.argmax(silhouette_scores)]
    return best_k


def apply_hdbscan(
    X: pd.DataFrame, min_cluster_size: int = 10
) -> tuple[HDBSCAN, np.ndarray]:
    """
    apply **HDBSCAN**
    """
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1)
    y_pred = hdb.fit_predict(X)
    return hdb, y_pred


def evaluate_clustering(X: pd.DataFrame, labels: np.ndarray) -> float:
    """
    evaluating clustering performance using various metrics
    """
    n_labels = len(set(labels))
    if n_labels <= 1:
        print("Need more than 1 cluster")
        return None

    sil_score = silhouette_score(X, labels)

    print(f"silhouette = {sil_score}")
    
    return sil_score
