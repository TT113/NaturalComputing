import random
import math

functions = {
    "sum_of_squares": -1,
    "sin": 1,
    "easom" : [2, [-100, 100]],
    "polynom": 1,
    "zlosin": 1,
    "drop_wave": [2, [-5.12, 5.12]],
    "eggcrate": 2}


def get_function():
    keys = function.keys()
    n = random.randint(0, len(keys))
    return globals()[keys[n]]


def get_info(name):
    return functions[name]


def get_function(name):
    print(name)
    return globals()[name]


def sum_of_squares(x):
    sum = 0
    for i in range(0, len(x)):
        sum += x[i]**2
    return sum


def sin(x):
    return 10*math.cos(x)


def polynom(x):
    if type(x) is list:
        x = x[0]
    return x**3-4*x**2+x-4


def zlosin(x):
    if type(x) is list:
        x = x[0]
    return math.sin(x**2) / x - 1 / 4 * x


def eggcrate(x):
    return x[0]**2 + x[1]**2 + 25(math.sin(x[0])**2+math.sin(x[1]))


def easom(x):
    return -math.cos(x[0]) * math.cos(x[1])*math.exp(-(x[0]-math.pi)**2 - (x[1]-math.pi)**2)

def drop_wave(x):
    return -(1+math.cos(12*(x[0]**2+x[1]**2)**0.5))/(0.5*(x[0]**2+x[1]**2) + 2)

# def f1(x):
#     return -math.cos(x[0]) * math.sin(x[1]) * math.exp(-(x[0] - math.pi) ** 2 + ((x[1] - math.pi) ** 2))


