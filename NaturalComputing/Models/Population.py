import numpy as np


class Population(np.ndarray):

    def __new__(cls, essential_params):
        obj = np.asarray(essential_params).view(cls)
        return obj

#    def remove(self):



