import common

import cv2
import numpy as np
from functools import reduce
from functools import cmp_to_key

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
    real_avg_area = sum(map(lambda a: a.area, meaningful))/len(meaningful) if len(meaningful) != 0 else 0

    #average of bounding rect areas
    bound_avg_area = sum(map(lambda a: (a.bound[2]-a.bound[0])*(a.bound[3]-a.bound[1]), meaningful))/len(meaningful) if len(meaningful) != 0 else 0

    #list of the contours in the bound region
    in_bound = list(filter(lambda e: utils.point_in_rect(utils.rect_center(e.bound), bounds), meaningful))
    in_bound_avg_area = sum(map(lambda e: e.area, in_bound))/len(in_bound) if len(in_bound) != 0 else 0
    
    #list of the contours outside the bound region
    out_bound = list(filter(lambda e: not utils.point_in_rect(utils.rect_center(e.bound), bounds), meaningful))
    out_bound_avg_area = sum(map(lambda e: e.area, out_bound))/len(out_bound) if len(out_bound) != 0 else 0

    tot_in_bound = list(filter(lambda e: utils.rect_in_rect(e.bound, bounds), meaningful))
    tot_in_bound_area = sum([a.area for a in tot_in_bound]) /len(tot_in_bound) if len(tot_in_bound) != 0 else 0

    tot_out_bound = list(filter(lambda e: not utils.rect_in_rect(e.bound, bounds), meaningful))
    tot_out_bound_area = sum([a.area for a in tot_out_bound]) /len(tot_out_bound) if len(tot_out_bound) != 0 else 0
    
    #median of the (real) areas
    median_area = np.median(np.array(list(map(lambda a: a.area, meaningful))))

    #size of the sum of all areas
    baseline = reduce(lambda a, b: a + b, map(lambda a: a.area,  meaningful))

#TODO add a check for totally in area roi
    
 #       inp = cv2.circle(cv2.cvtColor(input, cv2.COLOR_GRAY2BGR), (int(c[0]), int(c[1])), 3, (0,255, 0), -3)
 #       cv2.imshow(str(name), inp)

    return baseline, lc, la, lm, dominance, total, c, bound_avg_area, real_avg_area, median_area, len(in_bound), len(in_bound)/lm, len(tot_in_bound), len(tot_in_bound)/lm, in_bound_avg_area, out_bound_avg_area, tot_in_bound_area, tot_out_bound_area

def get_score_string(b, lc, la, lm, d, t, c, baa, raa, ma, lib, plib, ltib, pltib, ibaa, obaa, tibaa, tobaa, n="Score"):
    """Returns the string representing the score results ready to print"""
    return """{}:
    baseline                    := {}
    len(contour)~(aprx)-(mean)  := {} ~ {} - {}
    dominance                   := {} / {} = {:.3f}
    baricenter                  := {}
    bound/real/media avg area   := {:.3f} {:.3f} {:.3f}
  > in bound                    := {} ({:.2f})
    tot in bound                := {} ({:.2f})
    in/out bound avg area       := {:.3f} / {:.3f} 
    total in/out bound avg area := {:.3f} / {:.3f}
""".format(n, b, lc, la, lm, d, t, d/t, c, baa, raa, ma, lib, plib, ltib, pltib, ibaa, obaa, tibaa, tobaa)

def _max_rank_heu(a, b):
    """Default heuristic taking the one with the highest ratio of contours in the bound"""
    ibr = a[11] - b[11]
    return ibr if ibr != 0 else (a[12]-a[13]) - (b[12]-b[13])

def _filter_band_dominance(input, base, uepsilon, lepsilon):
    DOM, TOT = 4, 5
    avg = sum([i[DOM]/i[TOT] for i in base]) /len(base)
    return list(filter(lambda i: avg * uepsilon > i[DOM]/i[TOT] > avg * lepsilon, input)), (avg*uepsilon, avg*lepsilon)

def _filter_insuff_contours(input, base, epsilon):
    MEAN_CONT = 3
    avg = sum([i[MEAN_CONT] for i in base]) / len(base)
    return list(filter(lambda i: i[MEAN_CONT] > avg * epsilon, input)), avg * epsilon 

def _filter_low_median_area(input, base, epsilon):
    MEDIAN = 9
    avg = sum([i[MEDIAN] for i in base]) / len(base)
    return list(filter(lambda i: i[MEDIAN] > avg * epsilon, input)), avg * epsilon 

def _filter_negative_area(input, epsilon):
    IBAA, OBAA = 14, 15
    PLIB = 11
    return list(filter(lambda i: i[IBAA]*i[PLIB]>i[OBAA]*(1-i[PLIB])+epsilon, input))

#heu ideas:
#   max in ratio
#   average in area must be big, but not too big
#   in/out area must be balanced
#   FIXME not worky
def rank(s, verbose=False):
    """Returns the highest ranking best fit given a given heuristic, (or the default)"""
    f = s

    ic, fk = _filter_insuff_contours(f, s, 0.2)
    if verbose:
        print("Removed {} match{} due to insufficient contours (< {:.2f})".format(len(f)-len(ic), "es" if len(f)-len(ic) > 1 else "", fk))
    if len(ic) > 0:
        f = ic
    elif verbose:
        print("Rollback due to 0 len")

    ic, fk = _filter_band_dominance(f, s, 1.4, 0.3)
    if verbose:
        print("Removed {} match{} due to band dominance ({:.2f} > d > {:.2f})".format(len(f)-len(ic), "es" if len(f)-len(ic) > 1 else "", fk[0], fk[1]))
    if len(ic) > 0:
        f = ic  
    elif verbose:
        print("Rollback due to 0 len")

    ic, fk = _filter_low_median_area(f, s, 0.6)
    if verbose:
        print("Removed {} match{} due to low median area (< {:.2f})".format(len(f)-len(ic), "es" if len(f)-len(ic) > 1 else "", fk))
    if len(ic) > 0:
        f = ic  
    elif verbose:
        print("Rollback due to 0 len")

    na = _filter_negative_area(f, 0)
    if verbose:
        print("Removed {} match{} due to negative area ratio".format(len(f)-len(na), "es" if len(f)-len(na) > 1 else ""))
    if len(na) > 0:
        f = na
    elif verbose:
        print("Rollback due to 0 len")

    f.sort(reverse=True, key=lambda e: e[11])
    return s.index(f[0])
