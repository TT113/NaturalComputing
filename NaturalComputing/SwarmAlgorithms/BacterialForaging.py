from ..Models import Entity
from ..Models import Population
import numpy as np


class Bacteria(Entity):

    def __init__(self, problem_size):
        super().__new__(self, np.random.rand(1, problem_size))
        self.fitness = 0

    # def __init__(self, health, fitness):
    #     super().__new__(self, np.random.rand(1, problem_size))


reproduction_steps = 1000


def bacterial_foraging_algorithm():
    best_bacteria = Bacteria(2)
    best_bacteria.print()
    return best_bacteria
