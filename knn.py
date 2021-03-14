import common

import cv2
import numpy as np

import random

import generator
import extract

GLYPHS = "1234567890"

def _labelize(labels):
    return {k: i for i, k in enumerate(list(labels))}
def _delabelize(labels):
    return {i: k for i, k in enumerate(list(labels))}

def _unpackage(train_set):
    data, labels = list(), list()
    for (l, d) in train_set:
        data.append(d)
        labels.append(l)
    return np.array(data), np.array(labels)

@common.showtime
def train(train_set, size, dump=None, load=None):
    """Trains a knn with the given train_set of samples.
        size is the size of each sample
        if dump is a file path (without estention) it'll save the trainset there
        if load is a file path (without estention) it'll load the trainset from there
            train_set will be ignored if these are defined
            they can't be defined both as it makes no sense
    returns the knn object
    """
    if dump is not None and load is not None:
        raise ValueError("You can't both dump and load as it doesn't make sense")

    if load is not None:
        with np.load("{}.npz".format(load)) as save:
            data, labels = save["data"], save["labels"]
    else:
        data, labels = _unpackage(train_set)

    data = data.reshape(-1, size[0]*size[1]).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(data, cv2.ml.ROW_SAMPLE, labels)

    if dump is not None:
        np.savez("{}.npz".format(dump), data=data, labels=labels)
    
    return knn

@common.showtime
def nearest(input, knn, k=5, glyphs=GLYPHS, verbose=False):
    """Given a knn object and an input mask will return the label of the mask for the curren knn training"""
    samp = np.array(input).reshape(1, input.shape[0]*input.shape[1]).astype(np.float32)
    print(samp.shape)
    ret, res, neigh, dist = knn.findNearest(samp, k=k)

    if verbose:
        print("r:{}, res:{}, neigh:{}, dist:{}".format(ret, res, neigh, dist))

    lbl = _delabelize(glyphs)[res[0][0]]

    return lbl

def get_train_set(size, glyphs=GLYPHS, verbose=False):
    """Will generate a sized trainset"""
    result = list()
    lbl = _labelize(glyphs)
    for i in range(size):
        glyph = random.choice(glyphs) #TODO remove this random 
        plate = generator.get_all_tables(glyph)[glyph]

        mask, _ = extract.get_optimal_mask(plate)

        result.append((lbl[glyph], mask))
        if verbose:
            print("{}/{}".format(i, size))
        
    return result, mask.shape


if __name__ == "__main__":
    gen = False # if True will calculate a new trainset
    if gen:
        s, size = get_train_set(200, verbose=True)
    else:
        s = None
        size = (1, 89590)

    k = train(s, size, load="data_set")
    print("Trained")
    
    res = 0
    TOT = 30
    for i in range(TOT):
        c = str(i%10)
        t, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c])

        r = nearest(t, k)
        print("{} {}".format(r, c))
        if r == c:
            res+=1
    print("Accuracy: {:.2f} ({}/{})".format(res/TOT, res, TOT))
