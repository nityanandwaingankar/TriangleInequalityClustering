import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class TIClustering:
    def __init__(self, threshold):
        self.threshold = threshold
        self.clusters = []

    def fit(self, data):
        visited = set()
        for i, point in enumerate(data):
            if i in visited:
                continue

            # Find the neighborhood of the current point
            neighbors = self._find_neighbors(point, data)

            # Check triangle inequality and form a cluster
            cluster = self._form_cluster(point, neighbors, data)
            self.clusters.append(cluster)

            # Mark all points in the cluster as visited
            visited.update(cluster)

    def _find_neighbors(self, point, data):
        distances = euclidean_distances([point], data)[0]
        return np.where(distances <= self.threshold)[0]

    def _form_cluster(self, point, neighbors, data):
        cluster = set()
        for neighbor in neighbors:
            satisfies_ti = True
            for other in neighbors:
                if neighbor != other:
                    d_pn = np.linalg.norm(data[neighbor] - point)
                    d_no = np.linalg.norm(data[neighbor] - data[other])
                    d_po = np.linalg.norm(data[other] - point)
                    if not (d_pn + d_no >= d_po):
                        satisfies_ti = False
                        break
            if satisfies_ti:
                cluster.add(neighbor)
        return cluster

    def get_clusters(self):
        return [list(cluster) for cluster in self.clusters]


# import pandas as pd

# Example usage
# if __name__ == "__main__":
# Load the Iris dataset
# iris = load_iris()
# data = iris.data  # Features (sepal length, sepal width, petal length, petal width)
# feature_names = iris.feature_names
# target_names = iris.target_names

# Initialize the clustering algorithm
# tic = TIClustering(threshold=2.0)

# # Fit the data
# tic.fit(data)

# # Output clusters
# clusters = tic.get_clusters()
# print("Clusters:", clusters)

# # Create a DataFrame for cluster assignments
# cluster_assignments = []
# for i, cluster in enumerate(clusters):
#     for point in cluster:
#         cluster_assignments.append([point, i])

# # Convert to DataFrame for easier analysis and display
# df = pd.DataFrame(cluster_assignments, columns=["Point Index", "Cluster ID"])

# # Display the cluster assignments as a table
# print("\nCluster Assignments Table:")
# print(df)

# # Plot the clusters using only the first two features (sepal length and sepal width)
# plt.figure(figsize=(10, 6))

# # Plot each cluster with different colors
# for cluster_id, cluster in enumerate(clusters):
#     cluster_data = data[cluster]
#     plt.scatter(
#         cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster_id}"
#     )

# # Adding titles and labels
# plt.title("Triangle Inequality Clustering of Iris Dataset")
# plt.xlabel("Sepal Length")
# plt.ylabel("Sepal Width")
# plt.legend()

# # Show the plot
# plt.show()
