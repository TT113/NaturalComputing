import numpy as np


class Population(np.ndarray):

    def __new__(cls, dimension1, dimension2):
        obj = np.zeros(shape=(dimension1, dimension2))
        return obj

    def __init__(self, dimension1, dimension2):
        super().__init__()

    def __str__(self):
        return super().__str__()

#    def remove(self):



