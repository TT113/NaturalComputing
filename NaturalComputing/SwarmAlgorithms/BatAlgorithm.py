import numpy as np
import random as random

from NaturalComputing.TestFunctions import get_function
from NaturalComputing.TestFunctions import get_info


# class Bat(Entity):
#
#     def __new__(cls, coordinates_in_space):
#         obj = Entity.__new__(cls, coordinates_in_space)
#         return obj
#
#     def __init__(self, coordinates_in_space):
#         self.health = 0.0
#         self.fitness = 0.0
#         self.cost = 0.0
#         self.calc_cost()
#
#     def __str__(self):
#         return super().__str__()
#
#     def calc_cost(self):
#         self.cost = searching_function(self)
#
#     def serialize(self):
#         return np.array2string(self, formatter={'float_kind':lambda x: "%.2f" % x})[1:-1] + ' ' + str(self.cost)

test_function_name = "drop_wave"
searching_function = get_function(test_function_name)
search_space_dimension = get_info(test_function_name)[0]
segment = get_info(test_function_name)[1]


population_size = 50
generations_count = 5
loudness = 0.25
pulse_rate = 0.5
frequency_min = 0
frequency_max = 2
accuracy = 10**(-4)

function_min = 0

iteration = 0

lb = np.zeros((search_space_dimension))
ub = np.zeros((search_space_dimension))
frequencies = np.zeros((population_size))
velocities = np.zeros((population_size, search_space_dimension))
solution = np.zeros((population_size, search_space_dimension))
fitness = np.zeros((population_size))
best_prev = np.zeros((search_space_dimension))
best = np.zeros((search_space_dimension))

s = np.zeros((population_size, search_space_dimension))

def best_bat():
    i, j = 0, 0
    for i in range(population_size):
        if fitness[i] < fitness[j]:
            j = i
    for i in range(search_space_dimension):
        best[i] = solution[j][i]
    f_min = fitness[j]

def simplebounds(val, lower, upper):
    if val < lower:
        val = lower
    if val > upper:
        val = upper
    return val

for i in range(search_space_dimension):
    lb[i] = segment[0]
    ub[i] = segment[1]

for i in range(population_size):
    frequencies[i] = 0
    for j in range (search_space_dimension):
        velocities[i][j] = 0
        solution[i][j] = lb[j] + (ub[j] - lb[j]) * random.uniform(0, 1)
    fitness[i] = searching_function(solution[i])
best_bat()
for t in range(generations_count):
    for i in range(population_size):
        frequencies[i] = frequency_min + (frequency_min - frequency_max) * random.uniform(0, 1)
        for j in range(search_space_dimension):
            velocities[i][j] = velocities[i][j] + (solution[i][j] - best[j]) * frequencies[i]
            s[i][j] = solution[i][j] + velocities[i][j]
            s[i][j] = simplebounds(s[i][j], lb[j], ub[j])
            if random.uniform(0, 1) > pulse_rate:
                for j in range(search_space_dimension):
                    s[i][j] = best[j] + 0.001 * random.gauss(0, 1)
                    s[i][j] = simplebounds(s[i][j], lb[j], ub[j])
            f_new = searching_function(s[i])
            if f_new <= fitness[i] and random.uniform(0, 1) < loudness:
                solution[i] = s[i]
                fitness[i] = f_new
            if f_new <= function_min:
                best_prev = best
                best = s[i]
                function_min = f_new
            iteration += 1
            # print(iteration, function_min)
        # if abs(searching_function(best) - searching_function(best_prev)) < accuracy:
        #     break



print("iterations:", iteration)
print("min:", function_min)












