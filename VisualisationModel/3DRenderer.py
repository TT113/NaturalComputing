from __future__ import print_function

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from matplotlib import cm

from NaturalComputing.TestFunctions import get_function
from NaturalComputing.TestFunctions import get_info

test_function_name = "drop_wave"
func = get_function(test_function_name)
segment = get_info(test_function_name)[1]


def generate(X, Y, phi):
    return np.array([[func([x,y]) for y in Y] for x in X])



dataset = open('../NaturalComputing/SwarmAlgorithms/bacterias_foraging.txt', 'r')
# dataset = open('../NaturalComputing/SwarmAlgorithms/bat_algorithm.txt', 'r')


ticks = dataset.readlines()
dim = int(ticks[0]) + 1

ticks = [[float(p) for p in tick.split(' ')[:-1]] for tick in ticks[1:]]
ticks = [[tuple(tick[i*dim:i*dim+dim]) for i in range(len(tick)//dim)]for tick in ticks]
bugs = 50
ticks_X = np.array([[[ticks[ptr][bug][0]] for bug in range(len(ticks[ptr]))] for ptr in range(len(ticks))]).reshape(len(ticks), len(ticks[0]),)
ticks_Y = np.array([[[ticks[ptr][bug][1]] for bug in range(len(ticks[ptr]))] for ptr in range(len(ticks))]).reshape(len(ticks), len(ticks[0]),)
ticks_Z = np.array([[[ticks[ptr][bug][2]] for bug in range(len(ticks[ptr]))] for ptr in range(len(ticks))]).reshape(len(ticks), len(ticks[0]),)

fig = plt.figure()
fig.canvas.set_window_title('3-D algorithm visualization')
fig.suptitle("Drop wave function")
ax = fig.add_subplot(111, projection='3d')


xs = np.arange(segment[0], segment[1], 0.2, dtype=float)
ys = np.arange(segment[0], segment[1], 0.2, dtype=float)
X, Y = np.meshgrid(xs, ys)

# np.lin

# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-1.1, 0.01)

# Begin plotting.
wframe = None
points = None
tstart = time.time()


current_ptr = 0


def inc_ptr():
    global current_ptr
    current_ptr = (current_ptr + 1) % len(ticks_X)


Z = generate(xs, ys, 0)

# Z_normalized = (Z-Z.min())/(Z.max()-Z.min())
#
# colors = cm.viridis(Z_normalized)
# rcount, ccount, _ = colors.shape

wframe = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True, shade=True)
prev_timestamp = time.time()
paths = None

while True:
    # curr_timestamp = time.time()
    # # print(curr_timestamp - prev_timestamp)
    # prev_timestamp = curr_timestamp
    # If a line collection is already remove it before drawing.
    # if wframe:
    #     ax.collections.remove(wframe)
    # if points:
    #     for p in points:
    #         p.remove()
    # ax.clear()
        # points = None
        # ax.collections.remove(points)
    if points != None:
        for p in points:
            p.remove()
    # if paths != None:
    #     paths.remove()

    points = plt.plot(ticks_X[current_ptr], ticks_Y[current_ptr], ticks_Z[current_ptr], 'ko')

    # plt.plot(VecStart_x + VecEnd_x, VecStart_y + VecEnd_y, VecStart_z + VecEnd_z)

    # paths = ax.scatter(ticks_X[current_ptr], ticks_Y[current_ptr], ticks_Z[current_ptr], color='g')
    plt.pause(0.0000000001)
    # plt.show()

    inc_ptr()



# print('Average FPS: %f' % (100 / (time.time() - tstart)))