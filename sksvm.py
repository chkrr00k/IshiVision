import cv2
import numpy as np

from sklearn.svm import SVC

import ocr
import neighbour


class SkSvmOCR(ocr.OCR):
    def __init__(self, dump=None, load=None, train_set=None, verbose=None):
        self.sksvm = self.__train(train_set, dump, load)
    
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
    
    def __train(self, train_set=None, dump=None, load=None):
        if load is not None:
            with np.load("{}.npz".format(load)) as save:
                data, labels = save["data"], save["labels"]
        else:
            data, labels = self.__unpackage(ocr.OCR.get_train_set(train_set))
            data = np.array([neighbour.clean2(d) for d in data])
            #data = [self.__hog(self.__deskew(neighbour.clean2(i))) for i in data]
            #data = np.float32(data).reshape(-1, 64)
        
        sksvm = SVC(kernel="linear", C=1)
        sksvm.fit(np.array([d.flatten() for d in data]), labels)
        
        if dump is not None:
            np.savez("{}.npz".format(dump), data=data, labels=labels)
            
        return sksvm

    def read(self, img):
        res = self.sksvm.predict([neighbour.clean2(img).flatten()])
        return ocr.OCR.delabelize(ocr.GLYPHS)[res[0]]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass
