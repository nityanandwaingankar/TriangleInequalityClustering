import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class TIClustering:
    def __init__(self, threshold):
        """
        Initialize the clustering algorithm with a distance threshold.
        :param threshold: Distance threshold for forming neighborhoods (Eps).
        """
        self.threshold = threshold
        self.clusters = []
        self.index = []  # k-Neighborhood index

    def fit(self, data):
        """
        Fit the clustering algorithm on the data.
        :param data: Input dataset as a NumPy array.
        """
        # Step 1: Choose a global reference point (e.g., origin)
        reference_point = data[0]  # np.zeros(data.shape[1])

        # Step 2: Compute distances to the reference point and sort the data
        distances_to_ref = euclidean_distances(data, [reference_point]).flatten()
        sorted_indices = np.argsort(distances_to_ref)
        sorted_data = data[sorted_indices]

        # Step 3: Construct k-neighborhood index
        self.index = self._build_k_neighborhood_index(sorted_data)

        # Step 4: Form clusters using the triangle inequality
        visited = set()
        for i, point in enumerate(sorted_data):
            if i in visited:
                continue

            # Find the neighborhood for the current point
            neighbors = self.index[i]

            # Form a cluster and mark neighbors as visited
            cluster = self._form_cluster(point, neighbors, sorted_data)
            self.clusters.append(cluster)
            visited.update(cluster)

    def _build_k_neighborhood_index(self, data):
        """
        Build the k-neighborhood index for the dataset.
        :param data: Input dataset, sorted by distance to the reference point.
        :return: List of neighborhoods for each point.
        """
        n_points = data.shape[0]
        index = []
        for i in range(n_points):
            neighborhood = self._find_neighbors(data[i], data)
            index.append(neighborhood)
        return index

    def _find_neighbors(self, point, data):
        """
        Find the Eps-neighborhood of a point using the triangle inequality.
        :param point: The query point.
        :param data: Sorted dataset.
        :return: Indices of points within the threshold distance.
        """
        distances = euclidean_distances([point], data).flatten()
        neighbors = []
        for j, d in enumerate(distances):
            if d <= self.threshold:
                neighbors.append(j)
        return neighbors

    def _form_cluster(self, point, neighbors, data):
        """
        Form a cluster from the given point and its neighbors.
        :param point: The query point.
        :param neighbors: Indices of potential neighbors.
        :param data: Dataset.
        :return: Cluster as a set of indices.
        """
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
        """
        Get the clusters formed by the algorithm.
        :return: List of clusters, where each cluster is a list of point indices.
        """
        return [list(cluster) for cluster in self.clusters]
