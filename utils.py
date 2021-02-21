import cv2
import numpy as np
import math

def straight(img):
    r, c = img.shape[:2]
    shift = math.tan(.15)*r/2
    src, dst = np.float32([[0, 0],[c, 0],[0, r]]), np.float32([[-shift,0],[c-shift, 0],[shift, r]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (r, c))
