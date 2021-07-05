import cv2
import numpy as np

from sklearn.naive_bayes import GaussianNB

import ocr
import neighbour


class GaussianNBOCR(ocr.OCR):
    def __init__(self, dump=None, load=None, train_set=None, verbose=None):
        self.gnb = self.__train(train_set, dump, load)
    
    def __unpackage(self, train_set):
        data, labels = list(), list()
        for (l, d) in train_set:
            data.append(d)
            labels.append(l)
        return np.array(data), np.array(labels)
    
    def __train(self, train_set=None, dump=None, load=None):
        if load is not None:
            with np.load("{}.npz".format(load)) as save:
                data, labels = save["data"], save["labels"]
        else:
            data, labels = self.__unpackage(ocr.OCR.get_train_set(train_set))
            data = np.array([neighbour.clean2(d) for d in data])
        
        gnb = GaussianNB(var_smoothing=1)
        gnb.fit(np.array([d.flatten() for d in data]), labels)
        
        if dump is not None:
            np.savez("{}.npz".format(dump), data=data, labels=labels)
            
        return gnb

    def read(self, img):
        res = self.gnb.predict([neighbour.clean2(img).flatten()])
        return ocr.OCR.delabelize(ocr.GLYPHS)[res[0]]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass
