import matplotlib.pyplot as plt

from abc_algorithm import ABC
from objection_function import Iris

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

import numpy as np

data = MinMaxScaler().fit_transform(load_iris()['data'])

objective_function = Iris(dim=16)

optimizer = ABC(obj_function=objective_function, colony_size=30,
                n_iter=300, max_trials=100)
optimizer.optimize()

values = np.array(optimizer.optimality_tracking)
itr = range(300)
plt.plot(itr, 1/values, lw=0.5)
plt.legend(loc='upper right')
plt.show()

