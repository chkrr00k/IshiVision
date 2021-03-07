import numpy as np
import cv2
from matplotlib import pyplot as plt

import maxima
import categorize as cat
import visual
import common

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

    lmx, groups = maxima.local(buck, popular, epsilon=50, distance=maxima.ma_dist_init(maxima.DEF_MALA_MATRIX))

    LIMIT = int((img_hsv.shape[0]*img_hsv.shape[1])/(len(groups)*4))
    result = dict()
    for c in lmx:

        m, points = visual.mask_from_group(groups[c], buck, img_hsv.shape[:-1])

        space = (*c, 255)
        if len(points) > LIMIT:
            result[space] = m

    return result


