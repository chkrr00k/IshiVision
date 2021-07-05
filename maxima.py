import numpy as np
import cv2

EPSILON = 100
DEF_MALA_MATRIX = np.array([[1/6, 0],[0, 1]])

def eu_dist(a, b):
    """Euclidean distance"""
    return sum([(int(aa)-int(bb))**2 for aa, bb in zip(a, b)])**.5
def ma_dist_init(mtx=None, self=True):
    """Mahalanobis distance"""
    if mtx is None:
        return eu_dist
    else:
        inv = np.linalg.inv(mtx)
        return lambda a, b: (np.dot(np.dot(np.array([int(aa)-int(bb) for aa, bb in zip(a, b)]), inv), np.array([[int(aa)-int(bb)] for aa, bb in zip(a, b)])))[0]**.5


#
def local(input, keys, epsilon=EPSILON, distance=eu_dist):
    """Compute the local maxima in a given input dictionary where 
    the keys are the sorted keys of the dict.
    Data must be sorted such as [x][y] = num with num sorted"""

    lmax, groups = list(), dict()

    lmax.append(keys[0])
    groups[keys[0]]=list()
    for k in keys[1:]:
        candidate = None
        candidate_dist = None
        compressor = None
        compressor_dist = None
        for l in lmax:
            d = distance(l, k)
            if d < epsilon:
                if compressor is None or d < compressor_dist:
                    compressor = l
                    compressor_dist = d
            else:
                if candidate is not None and candidate_dist < d:
                    candidate = l
                    candidate_dist = d
        if compressor is None:
            lmax.append(k)
            groups[k] = [k]
        else:
            groups[compressor].append(k)
    return lmax, groups

