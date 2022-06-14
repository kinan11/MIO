import matplotlib.pyplot as plt

import numpy as np
from abc_algorithm import ABC
from objective_function import SumOfSquaredErrors

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

data = MinMaxScaler().fit_transform(load_iris()['data'])

objective_function = SumOfSquaredErrors(dim=12, n_clusters=3, data=data)
optimizer = ABC(obj_function=objective_function, colony_size=30, n_iter=300, max_trials=100)
optimizer.optimize()

def decode_centroids(centroids, n_clusters, data):
    return centroids.reshape(n_clusters, data.shape[1])
  
centroids = dict(enumerate(decode_centroids(optimizer.optimal_solution.pos, n_clusters=3, data=data)))

def assign_centroid(centroids, point):
    distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
    return np.argmin(distances)
  
custom_tgt = []
for instance in data:
    custom_tgt.append(assign_centroid(centroids, instance))

colors = ['r', 'g', 'b']
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
for instance, tgt in zip(data, custom_tgt):
    ax.scatter(instance[0], instance[1], instance[2], s=50, edgecolor='w', alpha=0.5, color=colors[tgt])

for centroid in centroids:
    ax.scatter(centroids[centroid][0], centroids[centroid][1],centroids[centroid][2],  color='k', marker='x', lw=5, s=500)
plt.title('Partitioned Data found by ABC')
plt.show()
itr = range(len(optimizer.optimality_tracking))
val = optimizer.optimality_tracking
plt.figure(figsize=(10, 9))
plt.plot(itr, val)
plt.title('Sum of Squared Errors')
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.show()