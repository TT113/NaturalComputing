import random
import math

functions_name = ["Sum of squares", "Easom’s function"]


# calls function with corresponding index
def get_function(n = random.randint(0, len(functions_name) - 1)):
    print(functions_name[n])
    return globals()['f'+str(n)]


def f0(x):

    sum = 0
    for i in range(0, len(x)):
        sum += x[i]**2
    return sum

#
# def f1(x):
#     return -math.cos(x[0]) * math.sin(x[1]) * math.exp(-(x[0] - math.pi) ** 2 + ((x[1] - math.pi) ** 2))


