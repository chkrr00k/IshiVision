import common

import cv2
import numpy as np
from functools import reduce

import utils

def score(input, cont, bounds=0, prt=False, name=""):
    aprx = utils.reduce_sections_area(cont)
    lc, la = len(cont), len(aprx)
    dominance = cv2.countNonZero(input)
    total = input.shape[0]*input.shape[1]

    baricenters = list(map(utils.rect_center, map(lambda a: a.bound, aprx)))
    c = reduce(lambda a, b: [a[0]+b[0], a[1]+b[1]], baricenters)
    c = (c[0]/len(baricenters), c[1]/len(baricenters))

#TODO Area, Real Area, Avg area, distribution, centroid (?)

    if prt:
        print("""{} {}:
    len(contour)        := {}
    len(approximation)  := {}
    dominance           := {}
    total               := {}
    baricenter          := {}
""".format(name, input.shape, lc, la, dominance, total, c))
