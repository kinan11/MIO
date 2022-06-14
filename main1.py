import numpy
import ga
import pickle
import ANN
import random
import matplotlib.pyplot
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import math


dataset=load_iris()
data_inputs2 = numpy.array(dataset.data)
data_outputs = numpy.array(dataset.target)
features_STDs = numpy.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:,]

# f = open("dataset_features.pkl", "rb")
# data_inputs2 = pickle.load(f)
# f.close()
# features_STDs = numpy.std(a=data_inputs2, axis=0)

# data_inputs = data_inputs2[:, features_STDs>50]
# print(features_STDs)

# f = open("outputs.pkl", "rb")
# data_outputs = pickle.load(f)
# f.close()

num_generations = 20
mutation_percent = 10

x=[]
v=[]

for i in range(10):
    #Tworzenie populacji i predkosci
    initial_pop_weights = []
    initial_v = []

    HL1_neurons = 5
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))
    input_HL1_v = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))

    HL2_neurons = 4
    HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))
    HL1_HL2_v = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))

    output_neurons = 3
    HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))
    HL2_output_v = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))

    initial_pop_weights.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))
    initial_v.append(numpy.array([input_HL1_v, HL1_HL2_v, HL2_output_v]))

    pop_weights_mat = numpy.array(initial_pop_weights)
    pop_weights_vector = ga.mat_to_vector(pop_weights_mat)
    pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat) ## z tego korzystamy w SSN

    pop_v_mat = numpy.array(initial_v)
    pop_v_vector = ga.mat_to_vector(pop_v_mat)
    pop_v_mat = ga.vector_to_mat(pop_v_vector, pop_v_mat) ## wagi w tej samej postaci

    x.append(pop_weights_mat)
    v.append(pop_v_mat)

accuracies = []
gbest = [0,0]
pbest = [0,0]
c1=2.05
c2=2.05
m=0
xi= 2/math.fabs(2-(c1+c2)-math.sqrt(c1+c2)*(c1+c2)-4*(c1+c2))

generation = 0
while generation < num_generations and gbest[1]-m<5:
    print("Generation : ", generation)
    accuracies.append(m)
    acc = numpy.empty(shape=(len(x)))
    pbest[0]=0
    pbest[1]=0

    for i in range(len(x)):
        fitness = ANN.fitness(x[i], data_inputs, data_outputs)
        acc[i] = fitness[0]
        if acc[i]>gbest[1]:
             gbest[1] = acc[i]
             gbest[0] = i

        if acc[i] > pbest[1]:
            pbest[1] = acc[i]
            pbest[0] = i
    m=max(acc)
    generation+=1


    for i in range(len(x)):
        for j in range(len(x[i][0])):
            for k in range(len(x[i][0][j])):
                #v[i][0][j][k]+=(random.uniform(0,c1)*(x[pbest[0]][0][j][k]-x[i][0][j][k]) + random.uniform(0,c2)*(x[gbest[0]][0][j][k]-x[i][0][j][k])) #klasyczny
                #v[i][0][j][k] += xi*((random.uniform(0, c1) * (x[pbest[0]][0][j][k] - x[i][0][j][k]) + random.uniform(0, c2) * ( x[gbest[0]][0][j][k] - x[i][0][j][k])) ) # ze współczynnikiem scisku
                v[i][0][j][k] = random.uniform(0.4, 0.9)*v[i][0][j][k] + (random.uniform(0, c1) * (x[pbest[0]][0][j][k] - x[i][0][j][k]) + random.uniform(0, c2) * (x[gbest[0]][0][j][k] - x[i][0][j][k])) #inertia-weight
                x[i][0][j][k]+=v[i][0][j][k]

plt.figure(figsize=(9, 5))
plt.plot(range(generation),accuracies, linewidth=3, color="black")
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
plt.xticks(numpy.arange(0, generation+1, 1), fontsize=8)
plt.yticks(numpy.arange(0, 101, 5), fontsize=8)
plt.show()
print(accuracies)