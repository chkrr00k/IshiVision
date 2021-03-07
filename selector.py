import common

import cv2
import numpy as np
from functools import reduce

import utils

@common.showtime
def get_score(input, cont, bounds, epsilon=1):
    """Calculate an euristic to find the best picture
Returns:
    baseline: the summed area of all white pixels
    lc: the number of full contours
    la: the number of reduced contours
    lm: the number of significant contours (filtered with area > epsilon)
    dominance: the number of white pixel (not to be confused with the summed area)
    total: the total number of pixels in the image
    c: the centroid of the contours
    bound_avg_area: the average area of the bounding squares
    real_avg_area: the average real area of the contours
    median_area: the median real area
    len_in_bound: the number of the contours in the given bound
    percentage_in_bound: the percentage of the contours in given bound
    in_bound_avg_area: the average real area of the contours in bound
    out_bound_avg_area: the average real area of the contours outside bound
    """
    #approximated contourns
    aprx = utils.reduce_sections_area(cont)

    #length for utility
    lc, la = len(cont), len(aprx)

    #number of white pixels
    dominance = cv2.countNonZero(input)
    total = input.shape[0]*input.shape[1]

    #list of important (area bigger than epsilon) contourns
    meaningful = list(filter(lambda a: a.area > epsilon, aprx))
    lm = len(meaningful)

    #centers of all rectangles
    baricenters = list(map(utils.rect_center, map(lambda a: a.bound, meaningful)))
    c = reduce(lambda a, b: [a[0]+b[0], a[1]+b[1]], baricenters)
    c = (c[0]/len(baricenters), c[1]/len(baricenters)) #actual centroid

    #average of the real areas
    real_avg_area = reduce(lambda a, b: a + b, map(lambda a: a.area, meaningful))/len(meaningful)

    #average of bounding rect areas
    bound_avg_area = reduce(lambda a, b: a + b, map(lambda a: (a.bound[2]-a.bound[0])*(a.bound[3]-a.bound[1]), meaningful))/len(meaningful)

    #list of the contours in the bound region
    in_bound = list(filter(lambda e: utils.point_in_rect(utils.rect_center(e.bound), bounds), meaningful))
    in_bound_avg_area = reduce(lambda a, b: a + b, map(lambda e: e.area, in_bound))/len(in_bound)
    
    #list of the contours outside the bound region
    out_bound = list(filter(lambda e: not utils.point_in_rect(utils.rect_center(e.bound), bounds), meaningful))
    out_bound_avg_area = reduce(lambda a, b: a + b, map(lambda e: e.area, out_bound))/len(out_bound)

    #median of the (real) areas
    median_area = np.median(np.array(list(map(lambda a: a.area, meaningful))))

    #size of the sum of all areas
    baseline = reduce(lambda a, b: a + b, map(lambda a: a.area,  meaningful))
    
 #       inp = cv2.circle(cv2.cvtColor(input, cv2.COLOR_GRAY2BGR), (int(c[0]), int(c[1])), 3, (0,255, 0), -3)
 #       cv2.imshow(str(name), inp)

    return baseline, lc, la, lm, dominance, total, c, bound_avg_area, real_avg_area, median_area, len(in_bound), len(in_bound)/lm, in_bound_avg_area, out_bound_avg_area

def get_score_string(b, lc, la, lm, d, t, c, baa, raa, ma, lib, plib, ibaa, obaa, n="Score"):
    """Returns the string representing the score results ready to print"""
    return """{}:
    baseline            := {}
    len(contour)~(aprx) := {} ~ {}
    len(meaningful)     := {}
    dominance           := {} / {}
    baricenter          := {}
    bound avg area      := {:.3f}
    real avg area       := {:.3f}
    median real area    := {:.3f}
max > in bound          := {} ({:.2f})
    in/out bound avg area   := {:.3f} / {:.3f} [Note: summed values]
""".format(n, b, lc, la, lm, d, t, c, baa, raa, ma, lib, plib, ibaa, obaa)

def rank(s):
    print("NO HEURISTIC CURRENTLY APPLIED")
    return np.argmax([i[11] for i in s])
