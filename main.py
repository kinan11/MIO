import ANN
import numpy as np
import matplotlib.pyplot as plt
from ArtificialBeeColony.abc_algorithm import ABC
from ArtificialBeeColony.objection_function import Iris
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

dataset=load_iris()
data_inputs = np.array(dataset.data)
data_outputs = np.array(dataset.target)
features_STDs = np.std(a=data_inputs, axis=0)

data = MinMaxScaler().fit_transform(load_iris()['data'])

objective_function = Iris(dim=16)

optimizer = ABC(obj_function=objective_function, colony_size=30, n_iter=300, max_trials=100)
optimizer.optimize()
print("Wagi optymalnego rozwiązania: ",optimizer.optimal_solution.pos.reshape(1,1,4,4))

acc, predictions = ANN.predict_outputs(optimizer.optimal_solution.pos.reshape(1,1,4,4)[0], data_inputs, data_outputs)
print("Najlepsza dokładność : ", acc)

values = 1/np.array(optimizer.optimality_tracking)
values = np.insert(values,0,0)

plt.figure(figsize=(9, 5))
plt.plot(range(301), values, linewidth=3, color="black")
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
plt.xticks(np.arange(0, 300+1, 50), fontsize=8)
plt.yticks(np.arange(0, 101, 5), fontsize=8)
plt.show()
