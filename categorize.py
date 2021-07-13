import numpy as np
import cv2

def get_buckets_naive(input):
    """Divide the given input array into indices buckets of the same value"""
    result = dict()
    for x in range(len(input)):
        for y in range(len(input[0])):
            px = tuple((v for v in input[x, y]))
            if px in result:
                result[px].append((x, y))
            else:
                result[px] = [(x, y)]
    return result

def get_buckets(input):
    """Returns the buckets in an image"""
    result = dict()
    tmp = None
    for c, v in np.ndenumerate(input):
        if c[2] == 0:
            tmp = v
        else:
            tmp = (tmp, v)
            if tmp in result:
                result[tmp].append((c[0], c[1]))
            else:
                result[tmp] = [(c[0], c[1])]
    return result

def sort_buckets(input, key=None, reverse=True):
    """Sorts the buckets into popularity and return an array with all the keys"""
    if key is None:
        key = lambda e: len(input[e])
    result = [*input]
    result.sort(key=key, reverse=reverse)
    return result

def filter_nth_percentile_bucket(input, percentile):
    """Filter the buckets given the nth percentile of popularity"""
    prcnt = np.percentile([len(v) for _, v in input.items()], percentile)
    return dict(filter(lambda e: len(e[1]) >= prcnt, input.items()))

def reduce_2_chan(input):
    return input[:,:,:-1]
