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

def _deskew(img, affine_flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR):
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

def _hog(img, bin_n=16):
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
def train(train_set=None, size=None, dump=None, load=None):
    """Trains a svm with the given train_set of samples.
        size is the size of each sample
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
        data, labels = _unpackage(train_set)
    
        data = [_hog(_deskew(i)) for i in data]

        data = np.float32(data).reshape(-1, 64)
    #     data = data.reshape(-1, size[0]*size[1]).astype(np.float32)

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
def nearest(input, svm, glyphs=GLYPHS, verbose=False):
    """Given a svm object and an input mask will return the label of the mask for the curren svm training"""
    
    deskewed = _deskew(input)
    hogdata = _hog(deskewed)
    samp = np.float32(hogdata).reshape(-1, 64)
#     print(samp, samp.shape)
#     samp = np.array(deskewed).reshape(1, deskewed.shape[0]*deskewed.shape[1]).astype(np.float32)
#     print(samp, samp.shape)
    
    res = svm.predict(samp)[1]

    if verbose:
        print("r:{}, res:{}".format(ret, res))

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
        s, size = get_train_set(20, verbose=True)
        k = train(train_set=s, size=size, dump="data_set")
    else:
#         s = None
#         size = (1, 89590)
        k = train(load="data_set")
        
    print("Trained")
    
    res = 0
    TOT = 5
    for i in range(TOT):
        c = str(i%10)
        t, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c])

        r = nearest(t, k)
        print("{} {}".format(r, c))
        if r == c:
            res+=1
    print("Accuracy: {:.2f} ({}/{})".format(res/TOT, res, TOT))
