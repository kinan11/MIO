import numpy as np

class SumOfSquaredErrors():

    def __init__(self, dim, n_clusters, data):
        self.n_clusters = n_clusters
        self.centroids = {}
        self.data = data
        self.dim = dim
        self.minf = 0.0
        self.maxf = 1.0

    def custom_sample(self):
        return np.repeat(self.minf, repeats=self.dim) + np.random.uniform(low=0, high=1, size=self.dim) * np.repeat(self.maxf - self.minf, repeats=self.dim)

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def decode(self, x):
        centroids = x.reshape(self.n_clusters,  self.data.shape[1])
        self.centroids = dict(enumerate(centroids))
        
    def evaluate(self, x):
        self.decode(x)

        clusters = {key: [] for key in self.centroids.keys()}
        for instance in self.data:
            distances = [np.linalg.norm(self.centroids[idx] - instance) for idx in self.centroids]
            clusters[np.argmin(distances)].append(instance)

        sum_of_squared_errors = 0.0
        for idx in self.centroids:
            distances = [np.linalg.norm(self.centroids[idx] - instance) for instance in clusters[idx]]
            sum_of_squared_errors += sum(np.power(distances, 2))
        return sum_of_squared_errors