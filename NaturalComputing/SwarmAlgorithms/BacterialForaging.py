import numpy as np
import random as random
import math as math
import operator

from NaturalComputing.Models.Entity import Entity
from NaturalComputing.TestFunctions import get_function
from NaturalComputing.TestFunctions import get_info



class Bacteria(Entity):

    def __new__(cls, coordinates_in_space):
        obj = Entity.__new__(cls, coordinates_in_space)
        return obj

    def __init__(self, coordinates_in_space):
        self.health = 0.0
        self.fitness = 0.0
        self.cost = 0.0
        self.calc_cost()

    def __str__(self):
        return super().__str__()

    def calc_cost(self):
        self.cost = searching_function(self)

    def serialize(self):
        return np.array2string(self, formatter={'float_kind':lambda x: "%.2f" % x})[1:-1] + ' ' + str(self.cost)


test_function_name = "drop_wave"
searching_function = get_function(test_function_name)
search_space_dimension = get_info(test_function_name)[0]
segment = get_info(test_function_name)[1]

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


def attract_repel(bacteria, population, d_attr, w_attr, h_rep, w_rep):
    attract = compute_cell_interaction(bacteria, population, -d_attr, -w_attr)
    repel = compute_cell_interaction(bacteria, population, h_rep, -w_rep)
    return attract + repel


def compute_cell_interaction(bacteria, population, depth, width):
    result = 0.0
    for other_bacteria in population:
        diff = 0.0
        diff += sum(x**2 for x in bacteria - other_bacteria)
        result += depth * math.exp(width * diff)
    return result


def evaluate(bacteria, population, d_attr, w_attr, h_rep, w_rep):
    bacteria.calc_cost()
    bacteria.fitness = bacteria.cost + attract_repel(bacteria, population, d_attr, w_attr, h_rep, w_rep)


def tumble_cell(bacteria, search_space):
    step = generate_random_direction(search_space_dimension)
    location_in_space = np.zeros(search_space_dimension)
    for i in range(0, search_space_dimension):
        coordinate = bacteria[i] + step_size * step[i]
        if coordinate < search_space[i][0]:
            coordinate = search_space[i][0]
        elif coordinate > search_space[i][1]:
            coordinate = search_space[i][1]
        location_in_space[i] = coordinate
    return Bacteria(location_in_space)


def check_for_best_solution(bacteria, best_bacteria):
    if best_bacteria is None or bacteria.cost < best_bacteria.cost:
        best_bacteria = bacteria
    return best_bacteria


def chemotactic(population, search_space):
    best_bacteria = None
    for i in range(0, number_of_chemotactic_steps):
        moved_cells = []
        for bacteria in population:
            health = 0
            evaluate(bacteria, population, attractant_depth, attractant_signal_width, repellent_effect_height,
                     repellent_effect_width)
            best_bacteria = check_for_best_solution(bacteria, best_bacteria)
            health += bacteria.fitness
            for j in range(0, swimming_length):
                new_cell = tumble_cell(bacteria, search_space)
                evaluate(new_cell, population, attractant_depth, attractant_signal_width, repellent_effect_height,
                         repellent_effect_width)
                best_bacteria = check_for_best_solution(new_cell, best_bacteria)
                if new_cell.fitness > bacteria.fitness:
                    break
                bacteria = new_cell
                health += bacteria.fitness
            bacteria.health = health
            moved_cells.append(bacteria)
        population = moved_cells
        file_to_save.write(serialize_population(population))
        # print(str(best_bacteria.fitness) + " " + str(best_bacteria.cost))
    return [best_bacteria, population]


def reinit_population(population):
    for bacteria in population:
        bacteria.health = 0


def reproduction(population):
    best_bacterias = sorted(population, key=operator.attrgetter('health'), reverse=True)[:int(len(population) / 2)]
    return best_bacterias + best_bacterias


def elimination_dispersal(population, search_space):
    for i in range(0, number_of_bacteria_in_population):
        if random.uniform(0, 1) < elimination_dispersal_probability:
            new_coordinates = get_random_vector_in_space(search_space)
            for j in range(0, search_space_dimension):
                population[i][j] = new_coordinates[j]
            population[i].calc_cost()


def locate_new_population_randomly_in_space(search_space):
    population = []
    for i in range(0, number_of_bacteria_in_population):
        bacteria = Bacteria(get_random_vector_in_space(search_space))
        population.append(bacteria)
    return population


def serialize_population(population):
    str = ''
    for bacteria in population:
        str += bacteria.serialize() + ' '
    return str + '\n'

def bacterial_foraging_algorithm():
    file_to_save.write(str(search_space_dimension) + '\n')
    best_bacteria = None
    search_space = generate_space(np.array(segment), search_space_dimension)
    population = locate_new_population_randomly_in_space(search_space)
    file_to_save.write(serialize_population(population))
    for i in range(0, number_of_elimination_dispersal_events):
        for j in range(0, number_of_reproduction_steps):
            best_for_chemotactic = chemotactic(population, search_space)[0]
            best_bacteria = check_for_best_solution(best_for_chemotactic, best_bacteria)
            population = reproduction(population)
        elimination_dispersal(population, search_space)
    print(best_bacteria.cost)
    file_to_save.close()


file_to_save = open('bacterias_foraging.txt', 'w')
bacterial_foraging_algorithm()
