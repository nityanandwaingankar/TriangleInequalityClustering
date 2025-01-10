import time
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from TIClustering import TIClustering
import matplotlib.pyplot as plt
import pandas as pd


# Example usage
def main():
    # Load Iris dataset
    iris = load_iris()
    data = iris.data
    labels_true = iris.target

    # Standardize the dataset
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Initialize the clustering algorithm
    tic = TIClustering(threshold=2.0)

    # Fit the data
    tic.fit(data)
    # Output clusters
    clusters = tic.get_clusters()
    print("Clusters:", clusters)

    # Create a DataFrame for cluster assignments
    cluster_assignments = []
    for i, cluster in enumerate(clusters):
        for point in cluster:
            cluster_assignments.append([point, i])

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
    plt.title("Triangle Inequality Clustering of Iris Dataset")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
