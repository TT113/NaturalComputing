import numpy as np
import random as random

from NaturalComputing.TestFunctions import get_function
from NaturalComputing.TestFunctions import get_info

from NaturalComputing.Models.AlgorithmStopMode import AlgorithmStopMode

class BatAlgorithm:
    population_size = 35
    generations_count = 2000
    loudness = 0.5
    pulse_rate = 0.5
    frequency_min = 0
    frequency_max = 2
    function_min = 0

    def __init__(self, **kwargs):
        self.search_space_dimension = kwargs['dimension']
        self.searching_function = kwargs['searching_function']
        self.segment = kwargs['segment']
        if 'accuracy' in kwargs:
            self.mode = AlgorithmStopMode.FIXED_TARGET
            self.accuracy = kwargs['accuracy']
            self.optimum = kwargs['optimum']
        self.lb = np.zeros((self.search_space_dimension))
        self.ub = np.zeros((self.search_space_dimension))
        self.frequencies = np.zeros((self.population_size))
        self.velocities = np.zeros((self.population_size, self.search_space_dimension))
        self.solution = np.zeros((self.population_size, self.search_space_dimension))
        self.fitness = np.zeros((self.population_size))
        self.best_prev = np.zeros((self.search_space_dimension))
        self.best = np.zeros((self.search_space_dimension))
        self.alpha = 0.5

        self.s = np.zeros((self.population_size, self.search_space_dimension))
        self.iteration = 0
        self.optimize()

    def best_bat(self):
        i, j = 0, 0
        for i in range(self.population_size):
            if self.fitness[i] < self.fitness[j]:
                j = i
        for i in range(self.search_space_dimension):
            self.best[i] = self.solution[j][i]
        self.function_min = self.fitness[j]

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def serialize_population(self):
        str = ''
        for i in range(self.population_size):
            str += self.serialize(self.solution[i]) + ' '
        return str + '\n'

    def serialize(self, solution):
        return np.array2string(solution, formatter={'float_kind':lambda x: "%.2f" % x})[1:-1] + ' ' \
               + str(self.searching_function(solution))

    def optimize(self):
        file_to_save = open('bat_algorithm.txt', 'w')
        file_to_save.write(str(self.search_space_dimension) + '\n')
        for i in range(self.search_space_dimension):
            self.lb[i] = self.segment[0]
            self.ub[i] = self.segment[1]
        for i in range(self.population_size):
            self.frequencies[i] = 0
            for j in range (self.search_space_dimension):
                self.velocities[i][j] = 0
                self.solution[i][j] = self.lb[j] + (self.ub[j] - self.lb[j]) * random.uniform(0, 1)
            self.fitness[i] = self.searching_function(self.solution[i])
        self.best_bat()
        file_to_save.write(self.serialize_population())
        for t in range(self.generations_count):
            for i in range(BatAlgorithm.population_size):
                self.frequencies[i] = self.frequency_min + (self.frequency_max - self.frequency_min) * random.uniform(0, 1)
                for j in range(self.search_space_dimension):
                    self.velocities[i][j] = self.velocities[i][j] + (self.solution[i][j] - self.best[j]) * self.frequencies[i]
                    self.s[i][j] = self.solution[i][j] + self.velocities[i][j]
                    self.s[i][j] = self.simplebounds(self.s[i][j], self.lb[j], self.ub[j])
                    if random.uniform(0, 1) > self.pulse_rate:
                        for j in range(self.search_space_dimension):
                            self.s[i][j] = self.best[j] + self.alpha * random.gauss(0, 1)
                            self.s[i][j] = self.simplebounds(self.s[i][j], self.lb[j], self.ub[j])
                    f_new = self.searching_function(self.s[i])
                    if f_new <= self.fitness[i] and random.uniform(0, 1) < self.loudness:
                        self.solution[i] = self.s[i]
                        self.fitness[i] = f_new
                    if f_new <= self.function_min:
                        self.best_prev = self.best
                        self.best = self.s[i]
                        self.function_min = f_new
                        if self.mode == AlgorithmStopMode.FIXED_TARGET and \
                                        abs(self.optimum - self.function_min) < self.accuracy:
                            return [self.function_min, 1]
                self.iteration += 1
                file_to_save.write(self.serialize_population())
        print("iterations:", self.iteration)
        print("min:", self.function_min)
        file_to_save.close()
        return [self.function_min, -1]


test_function_name = "ackley"
searching_function = get_function(test_function_name)
search_space_dimension = get_info(test_function_name)[0]
# search_space_dimension = 1
# segment = [-10, 10]
segment = get_info(test_function_name)[1]
BatAlgorithm(dimension=search_space_dimension,
                      searching_function=searching_function,
                      segment=segment,
                      accuracy=10**-4,
                      optimum= get_info(test_function_name)[2]).optimize()

# accuracy = 10**-2
# optimum = -1
# alphas = []
# ii = 0
# #0.463984375
# #0.463984375
# #0.47671875
# for i in range(0,10):
#     ii+=1
#     print(ii)
#     print("====================================")
#     alpha_range = [0, 1]
#     iteration = 0
#     while iteration == 0 or abs(alpha_range[1] - alpha_range[0]) > 0.01:
#         alpha = (alpha_range[1] + alpha_range[0]) / 2
#         alpha_eps = alpha + 0.01
#         results1 = []
#         results2 = []
#         for i in range(0, 5):
#             iteration_count = BatAlgorithm(search_space_dimension, searching_function, segment, alpha, accuracy, optimum).optimize()[1]
#             print("calc1")
#             results1.append(iteration_count)
#             average1 = sum(results1) / float(len(results1))
#         for i in range(0, 5):
#             iteration_count = BatAlgorithm(search_space_dimension, searching_function, segment, alpha_eps, accuracy, optimum).optimize()[1]
#             print("calc2")
#             results2.append(iteration_count)
#             average2 = sum(results2) / float(len(results2))
#         if average1 < average2:
#             alpha_range[1] = alpha
#         else:
#             alpha_range[0] = alpha
#         iteration += 1
#         print(average1)
#         print(average2)
#         print(alpha_range)
#     alphas.append((alpha_range[0] + alpha_range[1])/2)
#
# print(sum(alphas)/len(alphas))

# results = []
# stuck = 0
# while len(results) < 10:
#     test_function_name = "rosenbrock"
#     searching_function = get_function(test_function_name)
#     search_space_dimension = get_info(test_function_name)[0]
#     # search_space_dimension = 10
#     segment = get_info(test_function_name)[1]
#     result = BatAlgorithm(dimension=search_space_dimension,
#                       searching_function=searching_function,
#                       segment=segment,
#                       accuracy=10**-4,
#                       optimum= get_info(test_function_name)[2]).optimize()
#     print(str(searching_function.calls))
#     if result[1] != -1:
#         print("success")
#         results.append(searching_function.calls)
#     else:
#         stuck += 1
# print(sum(results)/len(results))
# print("stuck:", stuck)






