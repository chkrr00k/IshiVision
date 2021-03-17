import common

import cv2
import numpy as np

import abc

class OCR(abc.ABC):
    @property
    def PASSED(self):
        return True
    @property
    def FAILED(self):
        return False

    @abc.abstractmethod
    def read(self, img, **params):
        pass

    @staticmethod
    def get_train_set(size):
        pass

class MockOCR(OCR):
    def __init__(self):
        pass

    def read(self, img):
        return OCR.PASSED

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass

if __name__ == "__main__":

    OCR.get_train_set(1)
    with MockOCR() as t:
        assert t.read(None), "Mock read should always succeed"
