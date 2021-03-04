import common

import cv2
import numpy as np
import time
import math

import extract
import utils
import maxima
import selecter

img = cv2.imread("ref/gen/4.jpg") #16 7 3 12 4 5 6 2 10

#weight are approximated                        ^^
img = cv2.bilateralFilter(img, 9, 125, 50)
print("Image of shape {}".format(img.shape))

r, c = img.shape[:2]
t0 = time.time()
ms = extract.get_masks(img)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

# Select interesting mask among the many found by the get_masks
imask, i_n = selecter.select_image(ms)

# Create Window
source_window = 'Source {}'.format(i_n)
cv2.namedWindow(source_window)
cv2.imshow(source_window, imask)
max_thresh = 255
thresh = 100 # initial threshold

# cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, contours.thresh_callback)
# contours.thresh_callback(neighbour.clean(imask), thresh)
contours.thresh_callback(imask, thresh)

cv2.imshow("Base image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
