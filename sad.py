import common

import cv2
import numpy as np

import ocr
import renderer

class SadOCR(ocr.OCR):
    def __init__(self, dump=None, load=None, train_set=None, verbose=None):
        if load is not None or dump is not None:
            raise ValueError("Dump and load are not supported for this ocr")
        self.verbose=verbose
        self.train_set = SadOCR.get_train_set()

    def __sad(self, input, tmpl):
        if common.debug:
            print("Input size: {}, template size: {}".format(input.shape, tmpl.shape))
        input_h, input_w = input.shape
        tmpl_h, tmpl_w = tmpl.shape
        result = None
        for x, y, h, w in [(x, y, h, w) for x, h in zip(range(input_w-tmpl_w), range(tmpl_w, input_w)) for y, w in zip(range(input_h-tmpl_h),  range(tmpl_h, input_h))]:
            cand = np.sum(np.abs(input[y:w, x:h] - tmpl))
            if result is None or cand < result:
                result = cand
        return result
        

    def read(self, img):
        h = dict()
        for g, ref in self.train_set:
            sad = self.__sad(img, ref)
            if g not in h or h[g] > sad:
                h[g] = sad
            if self.verbose:
                print("Scanned {} with score {}".format(g, sad))
        if common.debug:
            print("Result: {}".format(h))
        
        m = min(h.items(), key=lambda e: e[1])[0]
        return m

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass
    
    @staticmethod
    def get_train_set():
        c = [(k, v) for k, v in renderer.get_all_glyphs_refs(ocr.GLYPHS, fonts=[cv2.FONT_HERSHEY_SCRIPT_COMPLEX]).items()]
        c.extend([(k, v) for k, v in renderer.get_all_glyphs_refs(ocr.GLYPHS, fonts=[cv2.FONT_HERSHEY_SIMPLEX]).items()])
        return c

class SsdOCR(ocr.OCR):
    def __init__(self, dump=None, load=None, train_set=None, verbose=None):
        if load is not None or dump is not None:
            raise ValueError("Dump and load are not supported for this ocr")
        self.verbose=verbose
        self.train_set = SadOCR.get_train_set()

    def __ssd(self, input, tmpl):
        if common.debug:
            print("Input size: {}, template size: {}".format(input.shape, tmpl.shape))
        input_h, input_w = input.shape
        tmpl_h, tmpl_w = tmpl.shape
        result = None
        for x, y, h, w in [(x, y, h, w) for x, h in zip(range(input_w-tmpl_w), range(tmpl_w, input_w)) for y, w in zip(range(input_h-tmpl_h),  range(tmpl_h, input_h))]:
            cand = np.sum(np.square(input[y:w, x:h] - tmpl))
            if result is None or cand < result:
                result = cand
        return result
        

    def read(self, img):
        h = dict()
        for g, ref in self.train_set:
            sad = self.__ssd(img, ref)
            if g not in h or h[g] > sad:
                h[g] = sad
            if self.verbose:
                print("Scanned {} with score {}".format(g, sad))
        if common.debug:
            print("Result: {}".format(h))
        
        m = min(h.items(), key=lambda e: e[1])[0]
        return m

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass
    
    @staticmethod
    def get_train_set():
        c = [(k, v) for k, v in renderer.get_all_glyphs_refs(ocr.GLYPHS, fonts=[cv2.FONT_HERSHEY_SCRIPT_COMPLEX]).items()]
        c.extend([(k, v) for k, v in renderer.get_all_glyphs_refs(ocr.GLYPHS, fonts=[cv2.FONT_HERSHEY_SIMPLEX]).items()])
        return c
