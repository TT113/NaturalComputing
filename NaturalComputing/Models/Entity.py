from abc import ABCMeta, abstractmethod
import numpy as np


class Entity(np.ndarray):

    # @abstractmethod
    # def convert(self):
    #     pass

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __str__(self):
        return super().__str__()



