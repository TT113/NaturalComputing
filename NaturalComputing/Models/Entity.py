from abc import ABCMeta, abstractmethod
import numpy as np


class Entity(np.ndarray):

    def __new__(cls, essential_params):
        obj = np.asarray(essential_params).view(cls)
        return obj

    @abstractmethod
    def convert(self):
        pass

    def print(self):
        print(self)
