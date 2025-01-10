import time
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


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
    # Load Iris dataset
    iris = load_iris()
    data = iris.data
    labels_true = iris.target

    # Standardize the dataset
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    exec_time_kmeans, kmeans_model = measure_execution_time(kmeans.fit, data)
    labels_kmeans = kmeans_model.labels_

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    exec_time_dbscan, dbscan_model = measure_execution_time(dbscan.fit, data)
    labels_dbscan = dbscan_model.labels_

    # Metrics computation for K-Means
    ari_kmeans = compute_ari(labels_true, labels_kmeans)
    silhouette_kmeans = compute_silhouette(data, labels_kmeans)

    # Metrics computation for DBSCAN
    ari_dbscan = compute_ari(labels_true, labels_dbscan)
    silhouette_dbscan = compute_silhouette(data, labels_dbscan)

    # Print results
    print("K-Means:")
    print(f"ARI: {ari_kmeans}")
    print(f"Silhouette Score: {silhouette_kmeans}")
    print(f"Execution Time: {exec_time_kmeans} seconds")

    print("\nDBSCAN:")
    print(f"ARI: {ari_dbscan}")
    print(f"Silhouette Score: {silhouette_dbscan}")
    print(f"Execution Time: {exec_time_dbscan} seconds")


if __name__ == "__main__":
    main()
