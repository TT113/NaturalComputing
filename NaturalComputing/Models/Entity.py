from abc import ABCMeta, abstractmethod
import numpy as np


class Entity(np.ndarray):

    @abstractmethod
    def convert(self):
        pass

    def __new__(cls, params):
        obj = np.asarray(params).view(cls)
        return obj

    def __init__(self, params):
        super().__init__()

    def __str__(self):
        return super().__str__()


