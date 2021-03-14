import numpy as np
import cv2
from matplotlib import pyplot as plt

import maxima
import categorize as cat
import visual
import common
import utils
import selector

@common.showtime
def get_masks(img, remove_white=True):
    """Gets the masks from an image using the local maxima method"""

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,2] = 255

    img_red = cat.reduce_2_chan(img_hsv)

    buck = cat.get_buckets(img_red)

    old_l = len(buck)
     
    p_buck = cat.filter_nth_percentile_bucket(buck, 40)
    if len(p_buck) > 0:
        buck = p_buck

    #if present, gets rid of white
    if remove_white and (0, 0) in buck:
        del buck[(0,0)] 

    popular = cat.sort_buckets(buck)
    
    #previously 50
    lmx, groups = maxima.local(buck, popular, epsilon=40, distance=maxima.ma_dist_init(maxima.DEF_MALA_MATRIX))

    LIMIT = int((img_hsv.shape[0]*img_hsv.shape[1])/(len(groups)*4))
    result = dict()
    for c in lmx:

        m, points = visual.mask_from_group(groups[c], buck, img_hsv.shape[:-1])

        space = (*c, 255)
        if len(points) > LIMIT:
            result[space] = m

    return result

@common.showtime
def get_optimal_mask(img, verbose=False, stra=False, write=False, show=False):
    """Given an image, will return its optimal mask for the next step of the ocr"""
    #weight are approximated
    img = cv2.bilateralFilter(img, 9, 125, 50)
    if verbose:
        print("Image of shape {}".format(img.shape))

    ms = get_masks(img)

    scores = list()

    for n, m in ms.items():
        #straigten the chars (ocr step 1) not needed
        if stra:
            m = utils.straight(m)

        #finds the contourns
        _, con, hi = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if write:
            cv2.imwrite("ref/{}.jpg".format(n), m)
        
        #puts all the rectangles in a list
        rects, cont = utils.get_parent_contours(con, hi)

        bounds = (m.shape[0]//5, m.shape[1]//8, m.shape[0]*4//5, m.shape[1]*7//8)
        sc = selector.get_score(m, cont, bounds)
        if verbose:
            print(selector.get_score_string(*sc, n=n))
        scores.append(sc)
        
        if show:
            #compress the rectangles that are superimposed to eachothers to have only the main ones
            cv2.imshow("{} Mask with main contourns".format(n), visual.print_contours_bounding_rect(m, utils.reduce_sections(rects), bounds))

    i = selector.rank(scores, verbose=verbose)
    if verbose:
        print("Best fit found: {}".format(i))
    n, best_fit = list(ms.items())[i]
    return best_fit, n

