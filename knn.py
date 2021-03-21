import common

import cv2
import numpy as np

import random
from functools import reduce

import ocr

class KnnOCR(ocr.OCR):

    def __init__(self, train_set=None, dump=None, load=None, glyphs=ocr.GLYPHS, verbose=False):
        self.knn = self.__train(train_set, dump, load)
        self.verbose=False
        self.glyphs = glyphs

    def read(self, input, k=5):
        """Reads the number in the input image passed, k is the knn paramether"""
        return self.__nearest(input, self.knn, k, self.glyphs, verbose=self.verbose)

    def __unpackage(self, train_set):
        data, labels = list(), list()
        for (l, d) in train_set:
            data.append(d)
            labels.append(l)
        return np.array(data), np.array(labels)

    @common.showtime
    def __train(self, train_set=None, dump=None, load=None):
        """Trains a knn with the given train_set of samples.
            if dump is a file path (without estention) it'll save the trainset there
            if load is a file path (without estention) it'll load the trainset from there
                train_set will be ignored if these are defined
        returns the knn object
        """

        if load is not None:
            with np.load("{}.npz".format(load)) as save:
                data, labels = save["data"], save["labels"]
        else:
            data, labels = self.__unpackage(train_set)

        size = reduce(lambda a, b: a*b, data[0].shape)
        data = data.reshape(-1, size).astype(np.float32)

        knn = cv2.ml.KNearest_create()
        knn.train(data, cv2.ml.ROW_SAMPLE, labels)

        if dump is not None:
            np.savez("{}.npz".format(dump), data=data, labels=labels)
        
        return knn

    @common.showtime
    def __nearest(self, input, knn, k, glyphs, verbose=False):
        """Given a knn object and an input mask will return the label of the mask for the curren knn training"""
        samp = np.array(input).reshape(1, input.shape[0]*input.shape[1]).astype(np.float32)

        ret, res, neigh, dist = knn.findNearest(samp, k=k)

        if verbose:
            print("r:{}, res:{}, neigh:{}, dist:{}".format(ret, res, neigh, dist))

        lbl = ocr.OCR.delabelize(glyphs)[res[0][0]]

        return lbl

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass


if __name__ == "__main__":
    import generator
    import extract
    
    gen = False # if True will calculate a new trainset
    size = 1 #size of the trainset
    TOT = 30 #size of the testset
    if gen:
        d="data_set"
        size = 1
        s = ocr.OCR.get_train_set(size, verbose=True)
        l = None

        assert len(s)==len(ocr.GLYPHS)*size, "Must generate the correct number of element ({}, {})".format(len(s), len(ocr.GLYPHS)*size)
    else:
        d=None
        s = None
        l = "data_set"
    print("Trained")
    
    with KnnOCR(dump=d, load=l, train_set=s, verbose=True) as o:
        assert o is not None, "A new object must be created"
        res = 0
        for i in range(TOT):
            c = str(i%10)
            t, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c])

            r = o.read(t)

            assert r is not None, "ocr.read(...) must yield a result"
            print("{} {}".format(r, c))
            if r == c:
                res+=1
        print("Accuracy: {:.2f} ({}/{})".format(res/TOT, res, TOT))
