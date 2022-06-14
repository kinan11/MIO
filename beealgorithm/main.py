import matplotlib.pyplot as plt

from abc_algorithm import ABC
from objection_function import Iris
import ANN

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from sklearn.datasets import load_iris


dataset=load_iris()
data_inputs2 = np.array(dataset.data)
data_outputs = np.array(dataset.target)
features_STDs = np.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:,]

data = MinMaxScaler().fit_transform(load_iris()['data'])

objective_function = Iris(dim=16)

optimizer = ABC(obj_function=objective_function, colony_size=30, n_iter=300, max_trials=100)
optimizer.optimize()
print("Wagi optymalnego rozwiązania: ",optimizer.optimal_solution.pos.reshape(1,1,4,4))

acc, predictions = ANN.predict_outputs(optimizer.optimal_solution.pos.reshape(1,1,4,4)[0], data_inputs, data_outputs)
print("Najlepsza dokładność : ", acc)

values = 1/np.array(optimizer.optimality_tracking)
values = np.insert(values,0,0)
itr = range(301)
plt.plot(itr, values, lw=0.5)
plt.legend(loc='upper right')
plt.show()

