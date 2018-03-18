from NaturalComputing.Models.Entity import Entity
from NaturalComputing.TestFunctions import get_function
from NaturalComputing.Models.Population import Population


import numpy as np
import random as random
import math as math
import operator

best = 100000000000000000


class Bacteria(Entity):

    def __new__(cls, coordinates_in_space):
        obj = Entity.__new__(cls, coordinates_in_space)
        obj.health = 0.0
        obj.fitness = 0.0
        obj.cost = 0.0
        return obj

    def __str__(self):
        return super().__str__()

    def calc_cost(self):
        global best
        self.cost = searching_function(self)
        if self.cost < best:
            best = self.cost


search_space_dimension = 2

number_of_bacteria_in_population = 100

number_of_split = int(number_of_bacteria_in_population / 2)

number_of_elimination_dispersal_events = 5
number_of_reproduction_steps = 100
number_of_chemotaxis_steps = 100

elimination_dispersal_probability = 0.0001

swimming_length = 3
step_size = 0.6

eliminate_probability = 0.25
attractant_depth = 0.1
attractant_signal_depth = 0.1
attractant_signal_width = 0.1
repellent_effect_height = 0.1
repellent_effect_width = 10.0


searching_function = get_function()


def get_random_vector_in_space(search_space):
    vec = np.zeros(search_space.shape[0])
    for i in range(0, vec.shape[0]):
        vec[i] = search_space[i, 0] + (search_space[i, 1] - search_space[i, 0]) * random.uniform(0, 1)
    return vec


def generate_space(bounds, search_space_dimension):
    search_space = np.zeros(shape=(search_space_dimension, 2))
    for i in range(0, search_space_dimension):
        search_space[i] = bounds
    return search_space


def generate_random_direction(search_space_dimension):
    return get_random_vector_in_space(generate_space([-1, 1], search_space_dimension))


def get_initial_population(number_of_bacteria_in_population, search_space):
    population = []
    for i in range(0, number_of_bacteria_in_population):
        population.append(Bacteria(get_random_vector_in_space(search_space)))
    return population


def interaction_step(bacteria, population):
    atracctant = 0.0
    repelent = 0.0
    for i in range(0, number_of_bacteria_in_population):
        diff = 0
        for j in range(0, search_space_dimension):
            diff += (bacteria[j] - population[i][j])**2
        atracctant += -1.0 * attractant_depth * math.exp(-attractant_signal_width * diff)
        repelent += repellent_effect_height * math.exp(-repellent_effect_width * diff)
    bacteria.fitness = bacteria.cost + atracctant + repelent


def tumble_cell(bacteria, search_space):
    step = generate_random_direction(search_space_dimension)
    for i in range(0, search_space_dimension):
        coordinate = bacteria[i] + step_size * step[i]
        if coordinate < search_space[i][0]:
            coordinate = search_space[i][0]
        elif coordinate > search_space[i][1]:
            coordinate = search_space[i][1]
        bacteria[i] = coordinate
    bacteria.step = step
    return bacteria


def swim_step(new_cell, current_cell, search_space):
    for i in range(0, search_space_dimension):
        coordinate = new_cell[i] + step_size * new_cell.step
        if coordinate < search_space[i][0]:
            coordinate = search_space[i][0]
        elif coordinate > search_space[i][1]:
            coordinate = search_space[i][1]
        new_cell[i] = coordinate


def chemotaxis(population, search_space):
    last_fitness = 0
    for i in range(0, number_of_bacteria_in_population):
        interaction_step(population[i], population)
        new_cell = tumble_cell(population[i], search_space)
        new_cell.calc_cost()
        interaction_step(new_cell, population)
        for j in range(0, search_space_dimension):
            population[i][j] = new_cell[j]
        population[i].cost = new_cell.cost
        population[i].fitness = new_cell.fitness
        population[i].health += population[i].fitness
        for j in range(0, swimming_length):
            if new_cell.fitness < last_fitness:
                last_fitness = new_cell.fitness
                new_cell.calc_cost()
                interaction_step(new_cell, population)
                for j in range(0, search_space_dimension):
                    population[i][j] = new_cell[j]
                    population[i].cost = new_cell.cost
                    population[i].fitness = new_cell.fitness
                    population[i].health += population[i].fitness
            else:
                break


def reproduction(population):
    population.sort(key=operator.attrgetter('health'))
    k = len(population) - 1
    for i in range(0, number_of_split):
        for j in range(0, search_space_dimension):
            population[k][j] = population[i][j]
            k -= 1
    for i in range(0, search_space_dimension):
        population[i].health = 0


def elimination_dispersal(population, search_space):
    for i in range (0, number_of_bacteria_in_population):
        if random.uniform(0, 1) < elimination_dispersal_probability:
            new_coordinates = get_random_vector_in_space(search_space)
            for j in range(0, search_space_dimension):
                population[i][j] = new_coordinates[j]
            population[i].calc_cost()


def locate__new_population_randomly_in_space(search_space):
    population = []
    for i in range(0, number_of_bacteria_in_population):
        bacteria = Bacteria(get_random_vector_in_space(search_space))
        bacteria.calc_cost()
        population.append(bacteria)
    return population


def bacterial_foraging_algorithm():
    search_space = generate_space(np.array([-1, 1]), search_space_dimension)
    population = locate__new_population_randomly_in_space(search_space)
    for i in range(0, number_of_elimination_dispersal_events):
        for j in range(0, number_of_reproduction_steps):
            for k in range(0, number_of_chemotaxis_steps):
                chemotaxis(population, search_space)
                print(best)
            reproduction(population)
        elimination_dispersal(population, search_space)


bacterial_foraging_algorithm()