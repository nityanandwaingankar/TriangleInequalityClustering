import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from TIClustering import TIClustering
import numpy as np


# Function to compute Adjusted Rand Index (ARI)
def compute_ari(labels_true, labels_pred):
    """
    Computes the Adjusted Rand Index (ARI).

    Parameters:
        labels_true (array-like): Ground truth labels.
        labels_pred (array-like): Cluster labels predicted by the algorithm.

    Returns:
        float: ARI score.
    """
    return adjusted_rand_score(labels_true, labels_pred)


# Function to compute Silhouette Score
def compute_silhouette(data, labels_pred):
    """
    Computes the Silhouette Score.

    Parameters:
        data (array-like): Dataset used for clustering.
        labels_pred (array-like): Cluster labels predicted by the algorithm.

    Returns:
        float: Silhouette score.
    """
    return silhouette_score(data, labels_pred)


# Function to measure execution time
def measure_execution_time(clustering_method, *args, **kwargs):
    """
    Measures execution time of a clustering method.

    Parameters:
        clustering_method (callable): The clustering method to be timed.
        *args: Positional arguments for the clustering method.
        **kwargs: Keyword arguments for the clustering method.

    Returns:
        tuple: (execution_time, clustering_result)
    """
    start_time = time.time()
    result = clustering_method(*args, **kwargs)
    execution_time = time.time() - start_time
    return execution_time, result


# Example usage
def main():
    ################## 2d generated dataset ##################
    # Generate a synthetic 2-feature dataset
    np.random.seed(44)  # For reproducibility  #seed = [42, ]
    n_samples = 150

    # Create two clusters
    cluster_1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples // 2, 2))
    cluster_2 = np.random.normal(loc=[-2, -2], scale=0.5, size=(n_samples // 2, 2))

    # Combine the clusters into one dataset
    data = np.vstack([cluster_1, cluster_2])
    labels_true = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Standardize the dataset
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # TIClustering
    tic = TIClustering(threshold=1.4)  # threshold = [1.4, ]
    exec_time_tic, _ = measure_execution_time(tic.fit, data)
    clusters = tic.get_clusters()

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    exec_time_kmeans, kmeans_model = measure_execution_time(kmeans.fit, data)
    labels_kmeans = kmeans_model.labels_

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    exec_time_dbscan, dbscan_model = measure_execution_time(dbscan.fit, data)
    labels_dbscan = dbscan_model.labels_

    print("Clusters:", clusters)

    # Create flat labels for TIClustering
    labels_tic = [-1] * len(data)
    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            labels_tic[point] = cluster_id

    # Create a DataFrame for cluster assignments
    cluster_assignments = []
    for i, cluster in enumerate(clusters):
        for point in cluster:
            cluster_assignments.append([point, i])

    # Metrics computation for K-Means
    ari_kmeans = compute_ari(labels_true, labels_kmeans)
    silhouette_kmeans = compute_silhouette(data, labels_kmeans)

    # Metrics computation for DBSCAN
    ari_dbscan = compute_ari(labels_true, labels_dbscan)
    silhouette_dbscan = compute_silhouette(data, labels_dbscan)

    # Metrics computation for TIClustering
    ari_tic = compute_ari(labels_true, labels_tic)
    silhouette_tic = compute_silhouette(data, labels_tic)

    # Print results
    print("K-Means:")
    print(f"ARI: {ari_kmeans}")
    print(f"Silhouette Score: {silhouette_kmeans}")
    print(f"Execution Time: {exec_time_kmeans} seconds")

    print("\nDBSCAN:")
    print(f"ARI: {ari_dbscan}")
    print(f"Silhouette Score: {silhouette_dbscan}")
    print(f"Execution Time: {exec_time_dbscan} seconds")

    print("\nTIClustering:")
    print(f"ARI: {ari_tic}")
    print(f"Silhouette Score: {silhouette_tic}")
    print(f"Execution Time: {exec_time_tic} seconds")

    # Convert to DataFrame for easier analysis and display
    df = pd.DataFrame(cluster_assignments, columns=["Point Index", "Cluster ID"])

    # Display the cluster assignments as a table
    print("\nCluster Assignments Table:")
    print(df)

    # Plot the clusters using only the first two features (sepal length and sepal width)
    plt.figure(figsize=(10, 6))

    # Plot each cluster with different colors
    for cluster_id, cluster in enumerate(clusters):
        cluster_data = data[cluster]
        plt.scatter(
            cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster_id}"
        )

    # Adding titles and labels
    plt.title("Triangle Inequality Clustering of Synthetic Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
