import common

import cv2
import numpy as np

import ocr
import renderer

class AreaOCR(ocr.OCR):
    def __init__(self, dump=None, load=None, train_set=None, verbose=False):
        if dump is not None or load is not None:
            raise ValueError("Dump and load are not supported for this method")
        self.verbose = verbose
        self.train_set = AreaOCR.get_train_set()

    def read(self, img):
        h = dict()
        for g, ref in self.train_set:
            sc_in, sc_out = self.__subtract(img, ref)
            if self.verbose:
                print("Scanned {} with score {}/{}".format(g, sc_in, sc_out))
            #selecting the ones with bigger sc_in if it-s equal then the one with highst sc_out will be
            #selected to be less error prone
            if (g in h and (sc_in > h[g][0] or (sc_in == h[g][0] and sc_out > h[g][1]))) or (g not in h):
                h[g] = (sc_in, sc_out)
        if common.debug:
            print("Result: {}".format(h))
        
        m = max(h.items(), key=lambda e: e[1][0]-e[1][1])
        return m[0]

    def __subtract(self, a, b, g=None):

        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0])) #this is insanity
            if common.debug:
                print("Reshaped ref image")

        aa = cv2.bitwise_and(a, a, mask=b)
        ao = cv2.bitwise_and(a, a, mask=~b)
        if g is not None:
            cv2.imshow("aa:{}".format(g), aa)
            cv2.imshow("ao:{}".format(g), ao)
        return cv2.countNonZero(aa), cv2.countNonZero(ao)

    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        pass
    
    @staticmethod
    def get_train_set():
        c = [(k, v) for k, v in renderer.get_all_glyphs_refs(ocr.GLYPHS, fonts=[cv2.FONT_HERSHEY_SCRIPT_COMPLEX]).items()]
        c.extend([(k, v) for k, v in renderer.get_all_glyphs_refs(ocr.GLYPHS, fonts=[cv2.FONT_HERSHEY_SIMPLEX]).items()])
        return c

if __name__ == "__main__":
    assert len(AreaOCR.get_train_set()) == len(ocr.GLYPHS)*2, "Must generate the correct number of tables"

    import generator
    import extract
    
    TOT = 30
    res = 0
    with AreaOCR(train_set=1) as o:
        for i in range(TOT):
            c = str(i%10)
            t, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c])
            r = o.read(t)
            print(type(r))
            print(type(c))
            print(r==c)
            if r == c:
                res =+ 1
            print("{} {}".format(c, r))
        print(res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
