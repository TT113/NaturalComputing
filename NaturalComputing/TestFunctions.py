import random
import math

functions = {
    "sum_of_squares": [-1, [-10, 10], 0],
    "sin": [2, [-100, 100], -1],
    "easom" : [2, [-100, 100]],
    "sphere" : [2, [-5.12, 5.12], 0],
    "polynom": [-1, [-10, 10], 0],
    "rosenbrock": [2, [-2.048, 2.048], 0],
    "zlosin": [1, [-10, 10], -10],
    "drop_wave": [2, [-5.12, 5.12], -1],
    "eggcrate": [2, [-4, 4], 0],
    "ackley": [2, [-32, 32], 0]}


class FunctionWithCallCount(object):
    def __init__(self, function):
        self.function = function
        self.calls = 0
        self.min = 0
        self.accuracy = []

    def __call__(self, x):
        self.calls += 1
        return self.function(x)

def get_function():
    keys = function.keys()
    n = random.randint(0, len(keys))
    return FunctionWithCallCount(globals()[keys[n]])


def get_info(name):
    return functions[name]


def get_function(name):
    print(name)
    return FunctionWithCallCount(globals()[name])


def sum_of_squares(x):
    sum = 0
    for i in range(0, len(x)):
        sum += x[i]**2
    return sum


def sin(x):
    return 10*math.cos(x)


def sphere(x):
    return sum([x_i**2 for x_i in x])

def polynom(x):
    if type(x) is list:
        x = x[0]
    return x**3-4*x**2+x-4


def zlosin(x):
    return math.sin(x**2) / x - 1 / 4 * x


def eggcrate(x):
    return x[0]**2 + x[1]**2 + 25*(math.sin(x[0])**2+math.sin(x[1])**2)


def easom(x):
    return -math.cos(x[0]) * math.cos(x[1])*math.exp(-(x[0]-math.pi)**2 - (x[1]-math.pi)**2)

def drop_wave(x):
    return -(1+math.cos(12*(x[0]**2+x[1]**2)**0.5))/(0.5*(x[0]**2+x[1]**2) + 2)


def rosenbrock(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.

def ackley(x):
    firstSum = 0.0
    secondSum = 0.0
    for xx in x:
        firstSum += xx**2.0
        secondSum += math.cos(2.0*math.pi*xx)
    n = float(len(x))
    return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e



