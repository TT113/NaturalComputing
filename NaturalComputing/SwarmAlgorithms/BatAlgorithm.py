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


class BatAlgorithm:

    test_function_name = "drop_wave"
    searching_function = get_function(test_function_name)
    search_space_dimension = get_info(test_function_name)[0]
    segment = get_info(test_function_name)[1]



    population_size = 50
    generations_count = 1
    loudness = 0.25
    pulse_rate = 0.5
    frequency_min = 0
    frequency_max = 2
    accuracy = 10**(-4)

    iteration = 0

    function_min = 0

    lb = np.zeros((search_space_dimension))
    ub = np.zeros((search_space_dimension))
    frequencies = np.zeros((population_size))
    velocities = np.zeros((population_size, search_space_dimension))
    solution = np.zeros((population_size, search_space_dimension))
    fitness = np.zeros((population_size))
    best_prev = np.zeros((search_space_dimension))
    best = np.zeros((search_space_dimension))

    s = np.zeros((population_size, search_space_dimension))

    def best_bat(self):
        i, j = 0, 0
        for i in range(self.population_size):
            if self.fitness[i] < self.fitness[j]:
                j = i
        for i in range(self.search_space_dimension):
            self.best[i] = self.solution[j][i]
        f_min = self.fitness[j]

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
               + str(BatAlgorithm.searching_function(solution))

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
            print(self.solution[i])
            self.fitness[i] = BatAlgorithm.searching_function(self.solution[i])
        self.best_bat()
        file_to_save.write(self.serialize_population())
        for t in range(self.generations_count):
            for i in range(BatAlgorithm.population_size):
                self.frequencies[i] = self.frequency_min + (self.frequency_min - self.frequency_max) * random.uniform(0, 1)
                for j in range(self.search_space_dimension):
                    self.velocities[i][j] = self.velocities[i][j] + (self.solution[i][j] - self.best[j]) * self.frequencies[i]
                    self.s[i][j] = self.solution[i][j] + self.velocities[i][j]
                    self.s[i][j] = self.simplebounds(self.s[i][j], self.lb[j], self.ub[j])
                    if random.uniform(0, 1) > self.pulse_rate:
                        for j in range(self.search_space_dimension):
                            self.s[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)
                            self.s[i][j] = self.simplebounds(self.s[i][j], self.lb[j], self.ub[j])
                    f_new = BatAlgorithm.searching_function(self.s[i])
                    if f_new <= self.fitness[i] and random.uniform(0, 1) < self.loudness:
                        self.solution[i] = self.s[i]
                        self.fitness[i] = f_new
                    if f_new <= self.function_min:
                        best_prev = self.best
                        best = self.s[i]
                        self.function_min = f_new
                    self.iteration += 1
                file_to_save.write(self.serialize_population())

                    # print(iteration, function_min)
                # if abs(searching_function(best) - searching_function(best_prev)) < accuracy:
                #     break
        print("iterations:", self.iteration)
        print("min:", self.function_min)
        file_to_save.close()




BatAlgorithm().optimize()






