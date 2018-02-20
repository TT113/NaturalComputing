from NaturalComputing.Models.Entity import Entity
from NaturalComputing.TestFunctions import get_function
from NaturalComputing.Models.Population import Population

import numpy as np


class Bacteria(Entity):

    def __new__(cls, search_space_size):
        obj = Entity(np.random.rand(1, search_space_size))
        return obj

    def __init__(self, search_space_size):
        super().__init__()

    def __str__(self):
        return super().__str__()


number_of_bacteria_in_population = 100
chemotactic_steps = 100
reproduction_steps = 100
number_of_elimination_dispersal_events = 5
elimination_dispersal_probability = 0.0001

swimming_length = 0.0001


searching_function = get_function()


def generate_search_space(bounds, search_space_dimension):
    search_space = np.zeros(shape=(search_space_dimension, 2))
    for i in range(0, search_space_dimension):
        search_space[i] = bounds
    return search_space


def locate_population_randomly_in_space(population, search_space_dimension):
    for i in range(0, number_of_bacteria_in_population):
        population[i] = Bacteria(search_space_dimension)
        population[i].cost = searching_function(population[i])
        population[i].fitness = 0


def bacterial_foraging_algorithm(search_space_dimension = 2):
    search_space = generate_search_space(np.array([-1, 1], search_space_dimension))
    bacteria_population = Population(number_of_bacteria_in_population, search_space_dimension)
    locate_population_randomly_in_space(bacteria_population, search_space_dimension)




    cells = [{'vector': self.random_vector(self.search_space)} for x in range(self.pop_size)]
    return best_bacteria


bacterial_foraging_algorithm()