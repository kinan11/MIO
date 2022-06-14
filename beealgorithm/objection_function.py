import numpy as np

from scipy import optimize
from deap.benchmarks import schwefel

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass
import ANN
import numpy
from sklearn.datasets import load_iris


dataset=load_iris()
data_inputs = numpy.array(dataset.data)
data_outputs = numpy.array(dataset.target)
features_STDs = numpy.std(a=data_inputs, axis=0)

class Iris():

    def __init__(self, dim):
        self.dim = dim
        self.minf = 0
        self.maxf = 1

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def custom_sample(self):
        return np.repeat(self.minf, repeats=self.dim) + np.random.uniform(low=0, high=1, size=self.dim) *np.repeat(self.maxf - self.minf, repeats=self.dim)

    def evaluate(self, x):
        return 1/ANN.fitness(x.reshape(1,1,4,4), data_inputs, data_outputs)[0]