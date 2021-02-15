import numpy as np
import cv2

EPSILON = 100
DEF_MALA_MATRIX = np.array([[1/6, 0],[0, 1]])

def eu_dist(a, b):
    """Euclidean distance"""
    return sum([(int(aa)-int(bb))**2 for aa, bb in zip(a, b)])**.5
def ma_dist_init(mtx=None):
    """Mahalanobis distance"""
    if mtx is None:
        return eu_dist
    else:
        inv = np.linalg.inv(mtx)
        return lambda a, b: (np.dot(np.dot(np.array([int(aa)-int(bb) for aa, bb in zip(a, b)]), inv), np.array([[int(aa)-int(bb)] for aa, bb in zip(a, b)])))[0]**.5

#data must be sorted such as [x][y] = num with num sorted
def local(input, keys, epsilon=EPSILON, verbose=False, distance=eu_dist):
    """Compute the local maxima in a given input dictionary where 
    the keys are the sorted keys of the dict"""

    lmax, groups = list(), dict()

    lmax.append(keys[0])
    groups[keys[0]]=list()
    if verbose:
        print("\tAppended first iteration")
    for k in keys[1:]:
        if verbose:
            print("Currently sorting {}".format(k))
        compressed = False
        candidate = None
        candidate_dist = None
        compressor = None
        compressor_dist = None
        for l in lmax:
            if verbose:
                print("\tAnalyzing {}".format(l))
            d = distance(l, k)
            if d < epsilon:
                if verbose:
                    print("\t\tCompressed {} as {} [as distance: {}]".format(k, l, d))
                compressed = True
                if compressor is None or d < compressor_dist:
                    compressor = l
                    compressor_dist = d
            else:
                if candidate is not None and candidate_dist < d:
                    candidate = l
                    candidate_dist = d
        if not compressed:
            if verbose:
                print("\t\t\tAppended {} due to {} [as distance: {}]".format(k, candidate, candidate_dist))
            lmax.append(k)
            groups[k] = list()
        else:
            if verbose:
                print("\t\t\tGrouped {} in {}".format(k, compressor))
            groups[compressor].append(k)
    for l in lmax:
        groups[l].append(l)
    return lmax, groups

#TODO remove all below here####################################################
#print(ma_dist_init(DEF_MALA_MATRIX)((0, 0), (10, 0)))
#print(ma_dist_init(DEF_MALA_MATRIX)((0, 0), (0, 10)))
#print(ma_dist_init(DEF_MALA_MATRIX)((0, 0), (10, 10)))
#print(ma_dist_init(DEF_MALA_MATRIX)((0, 0), (-10, 0)))

def compressed_2d_range(input):
    x = [x for x, _ in input]
    y = [y for _, y in input]
#    print("decompressing {},{}".format(x, y))
# consider a trapezoid a, b, c, d containing all data, to get only the significant part
# we have to compute, given x all the x's and y all the y
# (min(x), min(y))-------------(max(x), min(y))
#       |                               |
#       |                               |
# (min(x), max(y))-------------(max(x), max(y))
# 
#
    return (min(x), min(y)), (max(x), max(y))

def rect_intersect(rec1, rec2):
    rec1_h, rec1_w = rec1[2] - rec1[0], rec1[3] - rec1[1]
    rec2_h, rec2_w = rec2[2] - rec2[0], rec2[3] - rec2[1]
    x = max(rec1[0], rec2[0])
    y = max(rec1[1], rec2[1])
    w = abs(min(rec1[0] + rec1_w, rec2[0] + rec2_w) - x)
    h = abs(min(rec1[1] + rec1_h, rec2[1] + rec2_h) - y)
    #print("x: {} y: {} w: {} h: {}".format(x, y, w, h))
    return (x, y, x+w, y+h) if w > 0 and h > 0 else None

def rect_subtract(base, sub):
    return list(filter(lambda e: e[2] - e[0] > 0 and e[3] - e[1] > 0,
            
            [(base[0], base[1], sub[0], sub[1]), #c
            (base[0], sub[1], sub[0], sub[3]), 
            (base[0], sub[3], sub[0], base[3]), 
            (sub[0], sub[3], sub[2], base[3]), 
            (sub[2], sub[3], base[2], base[3]), 
            (sub[2], sub[1], base[2], sub[3]), 
            (sub[2], base[1], base[2], sub[1]), 
            (sub[0], base[1], sub[2], sub[1])]))
    
#print(rect_intersect((0, 1, 60, 107), (150, 1, 150, 1)))
#print(rect_subtract((0, 1, 6, 7), rect_intersect((1, 2, 3, 4), (0, 1, 6, 7))))
#print(rect_subtract((0, 1, 60, 107), rect_intersect((0, 1, 60, 107), (6, 57, 31, 163))))
