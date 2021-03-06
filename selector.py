import common

import cv2
import numpy as np
from functools import reduce

import utils

def score(input, cont, bounds, prt=False, name=""):
    aprx = utils.reduce_sections_area(cont)
    lc, la = len(cont), len(aprx)
    dominance = cv2.countNonZero(input)
    total = input.shape[0]*input.shape[1]

    baricenters = list(map(utils.rect_center, map(lambda a: a.bound, aprx)))
    c = reduce(lambda a, b: [a[0]+b[0], a[1]+b[1]], baricenters)
    c = (c[0]/len(baricenters), c[1]/len(baricenters))

    real_avg_area = reduce(lambda a, b: a + b, map(lambda a: a.area, aprx))/len(aprx)
    bound_avg_area = reduce(lambda a, b: a + b, map(lambda a: (a.bound[2]-a.bound[0])*(a.bound[3]-a.bound[1]), aprx))/len(aprx)

    in_bound = list(filter(lambda e: utils.point_in_rect(utils.rect_center(e), bounds), map(lambda e: e.bound, aprx)))
    if len(in_bound) == 0:
        print([utils.rect_center(a.bound) for a in aprx])
        print(bounds)

#TODO distribution, centroid (?)

    if prt:
        print("""{} {}:
    len(contour)        := {}
    len(approximation)  := {}
    dominance           := {}
    total               := {}
    baricenter          := {}
    bound avg area      := {}
    real avg area       := {}
    in bound            := {} ({:.2f})
""".format(name, input.shape, lc, la, dominance, total, c, bound_avg_area, real_avg_area, len(in_bound), len(in_bound)/la))

 #       inp = cv2.circle(cv2.cvtColor(input, cv2.COLOR_GRAY2BGR), (int(c[0]), int(c[1])), 3, (0,255, 0), -3)
 #       cv2.imshow(str(name), inp)


