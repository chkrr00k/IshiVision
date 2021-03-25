import common

import cv2
import numpy as np

import random
import abc

import generator
import extract

GLYPHS = "1234567890"

class OCR(abc.ABC):
    @abc.abstractmethod
    def read(self, img, **params):
        pass
    
    @staticmethod
    def labelize(labels):
        return {k: i for i, k in enumerate(list(labels))}
    @staticmethod
    def delabelize(labels):
        return {i: k for i, k in enumerate(list(labels))}

    @staticmethod
    def get_train_set(size, glyphs=GLYPHS, verbose=False):
        """Will generate a sized trainset"""
        result = list()
        lbl = OCR.labelize(glyphs)
        for i in range(size):
            plates = generator.get_all_tables(glyphs)
            for gl, plate in plates.items():
                mask, _ = extract.get_optimal_mask(plate)

                result.append((lbl[gl], mask))
                if verbose:
                    print("Appended generated glyph: '{}'".format(gl))
            if verbose:
                print("{}/{}".format(i+1, size))
            
        random.shuffle(result)
        return result


class MockOCR(OCR):
    def __init__(self, dump=None, load=None, train_set=None, verbose=None):
        import sys
        print("Warning! MockOCR has been selected", file=sys.stderr)

    def read(self, img):
        return "&"

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass

if __name__ == "__main__":

    size=1
    s = OCR.get_train_set(size)
    assert len(s) == len(GLYPHS)*size, "Must generate the correct number of elements"

    with MockOCR() as t:
        assert t.read(None), "Mock read should always succeed"
