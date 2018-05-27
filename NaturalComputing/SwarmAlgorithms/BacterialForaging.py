import numpy as np
import random as random
import math as math
import operator

from NaturalComputing.Models.Entity import Entity
from NaturalComputing.TestFunctions import get_function
from NaturalComputing.TestFunctions import get_info


class Bacteria(Entity):

    def __new__(cls, coordinates_in_space, searching_function):
        obj = Entity.__new__(cls, coordinates_in_space)
        return obj

    def __init__(self, coordinates_in_space, searching_function):
        self.health = 0.0
        self.fitness = 0.0
        self.cost = 0.0
        self.calc_cost(searching_function)

    def __str__(self):
        return super().__str__()

    def calc_cost(self, searching_function):
        self.cost = searching_function(self)

    def serialize(self):
        return np.array2string(self, formatter={'float_kind':lambda x: "%.2f" % x})[1:-1] + ' ' + str(self.cost)


class BacterialForaging:


    number_of_bacteria_in_population = 50

    number_of_split = int(number_of_bacteria_in_population / 2)

    number_of_elimination_dispersal_events = 3
    number_of_reproduction_steps = 3
    number_of_chemotactic_steps = 30

    elimination_dispersal_probability = 0.25

    swimming_length = 4
    step_size = 0.1

    attractant_depth = 0.1
    attractant_signal_depth = 0.1
    attractant_signal_width = 0.2
    repellent_effect_height = attractant_depth
    repellent_effect_width = 10.0

    def __init__(self, dimension, searching_function, segment):
        self.search_space_dimension = dimension
        self.searching_function = searching_function
        self.segment = segment

    def get_random_vector_in_space(self, search_space):
        vec = np.zeros(search_space.shape[0])
        for i in range(0, vec.shape[0]):
            vec[i] = search_space[i, 0] + (search_space[i, 1] - search_space[i, 0]) * random.uniform(0, 1)
        return vec

    def generate_space(self, bounds, search_space_dimension):
        search_space = np.zeros(shape=(search_space_dimension, 2))
        for i in range(0, search_space_dimension):
            search_space[i] = bounds
        return search_space

    def generate_random_direction(self, search_space_dimension):
        return self.get_random_vector_in_space(self.generate_space([-1, 1], search_space_dimension))


    def get_initial_population(self, number_of_bacteria_in_population, search_space):
        population = []
        for i in range(0, number_of_bacteria_in_population):
            population.append(Bacteria(self.get_random_vector_in_space(search_space), self.searching_function))
        return population

    def attract_repel(self, bacteria, population, d_attr, w_attr, h_rep, w_rep):
        attract = self.compute_cell_interaction(bacteria, population, -d_attr, -w_attr)
        repel = self.compute_cell_interaction(bacteria, population, h_rep, -w_rep)
        return attract + repel

    def compute_cell_interaction(self, bacteria, population, depth, width):
        result = 0.0
        for other_bacteria in population:
            diff = 0.0
            diff += sum(x**2 for x in bacteria - other_bacteria)
            result += depth * math.exp(width * diff)
        return result

    def evaluate(self, bacteria, population, d_attr, w_attr, h_rep, w_rep):
        bacteria.calc_cost(self.searching_function)
        bacteria.fitness = bacteria.cost + self.attract_repel(bacteria, population, d_attr, w_attr, h_rep, w_rep)

    def tumble_cell(self, bacteria, search_space):
        step = self.generate_random_direction(self.search_space_dimension)
        location_in_space = np.zeros(self.search_space_dimension)
        for i in range(0, self.search_space_dimension):
            coordinate = bacteria[i] + self.step_size * step[i]
            if coordinate < search_space[i][0]:
                coordinate = search_space[i][0]
            elif coordinate > search_space[i][1]:
                coordinate = search_space[i][1]
            location_in_space[i] = coordinate
        return Bacteria(location_in_space, self.searching_function)

    def check_for_best_solution(self, bacteria, best_bacteria):
        if best_bacteria is None or bacteria.cost < best_bacteria.cost:
            best_bacteria = bacteria
        return best_bacteria

    def chemotactic(self, population, search_space):
        best_bacteria = None
        for i in range(0, self.number_of_chemotactic_steps):
            moved_cells = []
            for bacteria in population:
                health = 0
                self.evaluate(bacteria,
                              population,
                              self.attractant_depth,
                              self.attractant_signal_width,
                              self.repellent_effect_height,
                              self.repellent_effect_width)
                best_bacteria = self.check_for_best_solution(bacteria, best_bacteria)
                health += bacteria.fitness
                for j in range(0, self.swimming_length):
                    new_cell = self.tumble_cell(bacteria, search_space)
                    self.evaluate(new_cell,
                                  population,
                                  self.attractant_depth,
                                  self.attractant_signal_width,
                                  self.repellent_effect_height,
                                  self.repellent_effect_width)
                    best_bacteria = self.check_for_best_solution(new_cell, best_bacteria)
                    if new_cell.fitness > bacteria.fitness:
                        break
                    bacteria = new_cell
                    health += bacteria.fitness
                bacteria.health = health
                moved_cells.append(bacteria)
            population = moved_cells
            self.file_to_save.write(self.serialize_population(population))
            # print(str(best_bacteria.fitness) + " " + str(best_bacteria.cost))
        return best_bacteria, population

    def reinit_population(self, population):
        for bacteria in population:
            bacteria.health = 0

    def reproduction(self, population):
        best_bacterias = sorted(population, key=operator.attrgetter('health'), reverse=True)[:int(len(population) / 2)]
        return best_bacterias + best_bacterias

    def elimination_dispersal(self, population, search_space):
        for i in range(0, self.number_of_bacteria_in_population):
            if random.uniform(0, 1) < self.elimination_dispersal_probability:
                new_coordinates = self.get_random_vector_in_space(search_space)
                for j in range(0, self.search_space_dimension):
                    population[i][j] = new_coordinates[j]
                population[i].calc_cost(self.searching_function)

    def locate_new_population_randomly_in_space(self, search_space):
        population = []
        for i in range(0, self.number_of_bacteria_in_population):
            bacteria = Bacteria(self.get_random_vector_in_space(search_space), self.searching_function)
            population.append(bacteria)
        return population


    def serialize_population(self, population):
        str = ''
        for bacteria in population:
            str += bacteria.serialize() + ' '
        return str + '\n'

    def optimize(self):
        self.file_to_save = open('bacterias_foraging.txt', 'w')
        self.file_to_save.write(str(self.search_space_dimension) + '\n')
        best_bacteria = None
        search_space = self.generate_space(np.array(self.segment), self.search_space_dimension)
        population = self.locate_new_population_randomly_in_space(search_space)
        self.file_to_save.write(self.serialize_population(population))
        for i in range(0, self.number_of_elimination_dispersal_events):
            for j in range(0, self.number_of_reproduction_steps):
                best_for_chemotactic, population = self.chemotactic(population, search_space)
                best_bacteria = self.check_for_best_solution(best_for_chemotactic, best_bacteria)
                population = self.reproduction(population)
            self.elimination_dispersal(population, search_space)
        self.file_to_save.close()
        return best_bacteria.cost


test_function_name = "zlosin"
searching_function = get_function(test_function_name)
search_space_dimension = get_info(test_function_name)[0]
segment = get_info(test_function_name)[1]
BacterialForaging(search_space_dimension, searching_function, segment).optimize()