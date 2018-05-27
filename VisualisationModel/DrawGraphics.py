import pygame
import sys
import math as math

import numpy as np

from VisualisationModel.Color import Color
from VisualisationModel.ScreenConfig import ScreenConfig

from NaturalComputing.TestFunctions import get_function


dataset = open('../NaturalComputing/SwarmAlgorithms/bacterias_foraging.txt', 'r')

ticks = dataset.readlines()
dim = int(ticks[0]) + 1

ticks = [[float(p) for p in tick.split(' ')[:-1]] for tick in ticks[1:]]
ticks = [[tuple(tick[i*dim:i*dim+dim]) for i in range(len(tick)//dim)]for tick in ticks]

dataset.close()

pygame.init()
screen = pygame.display.set_mode((ScreenConfig.screen_width, ScreenConfig.screen_height))

def coord_to_screen_point(point):
    val =  -(ScreenConfig.viewport_center[0] - point[0]) * ScreenConfig.x_pixels_for_one + ScreenConfig.screen_width//2, \
           (ScreenConfig.viewport_center[1] - point[1]) * ScreenConfig.y_pixels_for_one + ScreenConfig.screen_height//2
    return val

def fill_background(surface):
    surface.fill(ScreenConfig.background_color)


def draw_net(surface, step=1):
    for x in range(ScreenConfig.bounding_box[0][0], ScreenConfig.bounding_box[1][0], step):
        pygame.draw.line(surface, Color.grey, coord_to_screen_point((x, ScreenConfig.bounding_box[0][1])),
                                              coord_to_screen_point((x, ScreenConfig.bounding_box[1][1])))
    for y in range(ScreenConfig.bounding_box[0][1], ScreenConfig.bounding_box[1][1], step):
        pygame.draw.line(surface, Color.grey, coord_to_screen_point((ScreenConfig.bounding_box[0][0], y)),
                                              coord_to_screen_point((ScreenConfig.bounding_box[1][0], y)))

def draw_function(f, surface):
    x_vals = np.arange(-100, 100, 0.01)
    y_vals = [f(x) for x in x_vals]

    for i in range(1, len(x_vals)):
        pygame.draw.line(surface, Color.green, coord_to_screen_point((x_vals[i-1], y_vals[i-1])),
                                              coord_to_screen_point((x_vals[i], y_vals[i])), 2)

class rendering_context:
    tick = 0
    tick_step = 10
    ticks_len = len(ticks)
    @staticmethod
    def next():
        # rendering_context.tick = min(rendering_context.ticks_len-1, (rendering_context.tick + rendering_context.tick_step))
        rendering_context.tick = (rendering_context.tick + rendering_context.tick_step) % rendering_context.ticks_len

def draw_bacterias():
    for i in ticks[rendering_context.tick]:
        coord = coord_to_screen_point(i)
        coord = int(coord[0]), int(coord[1])
        pygame.draw.circle(screen, Color.blue, coord, 3)


function = get_function('zlosin')
static = pygame.Surface((ScreenConfig.screen_width, ScreenConfig.screen_height))
fill_background(static)
draw_net(static)
draw_function(function, static)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        continue
    screen.blit(static, (0, 0))
    draw_bacterias()
    rendering_context.next()
    pygame.time.delay(100)
    pygame.display.update()