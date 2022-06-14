import numpy as np
import matplotlib.pyplot as plt
from ArtificialBeeColony.abc_algorithm import ABC
from ArtificialBeeColony.objective_function import SumOfSquaredErrors
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

def decode_centroids(centroids, n_clusters, data):
    return centroids.reshape(n_clusters, data.shape[1])

def assign_centroid(centroids, point):
    distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
    return np.argmin(distances)

data = MinMaxScaler().fit_transform(load_iris()['data'][:, [1,3]])

objective_function = SumOfSquaredErrors(dim=6, n_clusters=3, data=data)

optimizer = ABC(obj_function=objective_function, colony_size=30, n_iter=300, max_trials=100)

optimizer.optimize()

centroids = dict(enumerate(decode_centroids(optimizer.optimal_solution.pos, n_clusters=3, data=data)))

custom_tgt = []
for instance in data:
    custom_tgt.append(assign_centroid(centroids, instance))

colors = ['r', 'g', 'b']
plt.figure(figsize=(9,9))

for instance, tgt in zip(data, custom_tgt):
    plt.scatter(instance[0], instance[1], s=50, edgecolor='w',alpha=0.5, color=colors[tgt])
for centroid in centroids:
    plt.scatter(centroids[centroid][0], centroids[centroid][1],  color='k', marker='x', lw=5, s=500)

plt.title('Partitioned Data found by ABC')
plt.ylabel('SepalWidthCm')
plt.xlabel('PetalWidthCm')
plt.show()
plt.figure(figsize=(9,5))
plt.plot(range(len(optimizer.optimality_tracking)), optimizer.optimality_tracking, linewidth=3, color="black")
plt.title('Sum of Squared Errors')
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.show()