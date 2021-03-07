import cv2
import numpy as np

def create_maxima_map(buckets, maxima):
    """Visualize the maxima map given the bucket list and the actual maxima"""
    d = np.zeros((256, 256), np.uint8)
    for k, v in buckets.items():
        d[k[1], k[0]] = 127
    d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    for m in maxima:
        cv2.circle(d, m, 5, (0, 255, 255), 0)
    return d

def create_map(points):
    """Visualize the map of the given point"""
    se = np.zeros((256, 256), np.uint8)
    for p in points:
        se[p[1], p[0]] = 255
    return se

def mask_from_group(group, buckets, shape):
    """Creates the mask from group, the buckets and the actual image shape"""
    m = np.zeros(shape, np.uint8)
    points = [i for p in group for i in buckets[p]]
    for p in points:
        m[p[0], p[1]] = 255
    return m, points

def print_contours_bounding_rect(input, rects, bounds=None):
    """Returns the image in BGR space with the intresting area highlighted and all the rects shown"""
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)    
    for [x, y, xx, yy] in rects:
        cv2.rectangle(input, (x, y), (xx, yy), (255, 0, 0), 1)
    #actual intresting parts highlighted here
    if bounds:
        cv2.rectangle(input, bounds[:2], bounds[2:], (0, 255, 0), 1)
        cv2.line(input, ((input.shape[0]//2),0), (input.shape[0]//2, input.shape[1]), (0, 255, 0), 1)

    return input
