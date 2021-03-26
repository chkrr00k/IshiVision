#!/usr/bin/env python
# coding: utf-8

# In[17]:


import common

import cv2
import numpy as np

import random
#from functools import reduce

import ocr
import neighbour

class SvmOCR(ocr.OCR):

    def __init__(self, train_set=None, dump=None, load=None, glyphs=ocr.GLYPHS, verbose=False):
        self.verbose=verbose
        self.glyphs = glyphs
        self.svm = self.__train(train_set, dump, load)

    def read(self, input):
        """Reads the number in the input image passed"""
        return self.__nearest(neighbour.clean2(input), self.svm, self.glyphs, verbose=self.verbose)

    def __unpackage(self, train_set):
        data, labels = list(), list()
        for (l, d) in train_set:
            data.append(d)
            labels.append(l)
        return np.array(data), np.array(labels)
    
    #@common.showtime    
    def __deskew(self, img, affine_flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR):
        """Deskew an image if there is the necessity.
            img can be of any size
        returns the deskewed image
        """
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*img.shape[1]*skew], [0, 1, 0]]) # XXX or shape[0]
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=affine_flags)
        return img
    
    #@common.showtime
    def __hog(self, img, bin_n=16):
        """Create a histogram of 64 bits, with bins of 16, from the input image"""
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist

    @common.showtime
    def __train(self, train_set=None, dump=None, load=None):
        """Trains a svm with the given train_set of samples.
            if dump is a file path (without estention) it'll save the trainset there
            if load is a file path (without estention) it'll load the trainset from there
                train_set will be ignored if these are defined
                they can't be defined both as it makes no sense
        returns the svm object
        """
        
        if dump is not None and load is not None:
            raise ValueError("You can't both dump and load as it doesn't make sense")

        if load is not None:
            svm = cv2.ml.SVM_load("{}.dat".format(load))
        else:
            t = ocr.OCR.get_train_set(train_set, verbose=self.verbose)
            data, labels = self.__unpackage(t)
            
            data = [self.__hog(self.__deskew(neighbour.clean2(i))) for i in data]

            data = np.float32(data).reshape(-1, 64)

            svm = cv2.ml.SVM_create()
            svm.setKernel(cv2.ml.SVM_LINEAR)
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.setC(2.67)
            svm.setGamma(5.383)
            svm.train(data, cv2.ml.ROW_SAMPLE, labels)

        if dump is not None:
            svm.save('{}.dat'.format(dump))

        return svm

    @common.showtime
    def __nearest(self, input, svm, glyphs, verbose=False):
        """Given a svm object and an input mask will return the label of the mask for the curren svm training"""

        deskewed = self.__deskew(input)
        hogdata = self.__hog(deskewed)
        samp = np.float32(hogdata).reshape(-1, 64)

        res = svm.predict(samp)[1]

        if verbose:
            print("res:{}".format(res))

        lbl = ocr.OCR.delabelize(glyphs)[res[0][0]]

        return lbl

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass


if __name__ == "__main__":
    import generator
    import extract
    
    gen = True # if True will calculate a new trainset
    size = 1 #size of the trainset
    TOT = 30 #size of the testset
    if gen:
        d="data_set"
        s = ocr.OCR.get_train_set(size, verbose=True)
        l = None

        assert len(s)==len(ocr.GLYPHS)*size, "Must generate the correct number of element ({}, {})".format(len(s), len(ocr.GLYPHS)*size)
    else:
        d=None
        s = None
        l = "data_set"
    print("Trained")
    
    with SvmOCR(dump=d, load=l, train_set=size, verbose=True) as o:
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

