"""
File: clustering.py
Description: Apply clustering algorithms on data
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

def run_kmeans_with_best_k(data, method, best_k, random_state = 42):
    """
    Runs KMeans with the best k and returns cluster labels and centroids.
    """
    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)
    plt.figure(figsize=(7, 6))
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, s=40)
    plt.title(f"K-Means Clustering (k={best_k}) using {method}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()