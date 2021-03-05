#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math

import utils
import neighbour
import generator
import matcher

tables = generator.get_all_tables("1234567890", thic=20)

infos = matcher.get_infos(tables)


# In[2]:


def calculate_mean_contours_and_mean_area(ms):
    mean_contours = 0
    mean_area = 0

    for n, m in ms.items():
        #straigten the chars (ocr step 1)
#         mc = utils.straight(m)

#         mc = neighbour.clean(m)
        mc = m

        #finds the contourns
        _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #puts all the rectangles in a list
        rects = list()

        for c, [_, _, _, Pa] in zip(con, hi[0]):        
            re = cv2.boundingRect(c)
            if Pa < 0:
                mean_area = mean_area + cv2.contourArea(c)
                rects.append(re)
        mean_contours += len(utils.reduce_sections(rects))

    mean_area = mean_area / len(ms.items())
    mean_contours = mean_contours // len(ms.items())
    
    return mean_contours, mean_area

def select_image(ms):    
    max = 0
    min = math.inf
    mean_contours, mean_area = calculate_mean_contours_and_mean_area(ms)
    imask = None
    i_n = None
    
    for n, m in ms.items():
        r, c = m.shape[:2]
        
        #straigten the chars (ocr step 1)
#         mc = utils.straight(m)

#         mc = neighbour.clean(m)
        mc = m
        
        #selects the intresting part (the center)
        bndu, bndl = (r//8, c//7), (r*7//8, c*6//7)
        mc = mc[bndu[0]:bndl[0], bndu[1]:bndl[1]]
        
#         plt.imshow(mc, cmap='gray', vmin=0, vmax=255)
#         plt.show()
        
        #finds the contourns
        _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count pixels
        white_pixels = cv2.countNonZero(mc)
        black_pixels = (mc.shape[0] * mc.shape[1]) - white_pixels

        if white_pixels > black_pixels:
            # Invert the image pixels
            print('Inverted', n)
            mc = ~mc

        # Dilation
        if len(con) > mean_contours:
            print('Closed', n)
    #         dil_factor = int(math.floor(math.log10(abs(mc.shape[0])))) + 2
    #         el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2*dil_factor + 1), int(2*dil_factor + 1)))
    #         mc = cv2.dilate(mc, el)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#             kernel = np.ones((5,5),np.uint8)
            mc = cv2.morphologyEx(mc, cv2.MORPH_CLOSE, kernel)

            # Recalculate contours
            _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        else:
            print('Opened', n)
    #         dil_factor = int(math.floor(math.log10(abs(mc.shape[0])))) + 2
    #         el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2*dil_factor + 1), int(2*dil_factor + 1)))
    #         mc = cv2.dilate(mc, el)
#             kernel = np.ones((5,5),np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mc = cv2.morphologyEx(mc, cv2.MORPH_OPEN, kernel)

            # Recalculate contours
            _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Recount pixels
        white_pixels = cv2.countNonZero(mc)
        black_pixels = (mc.shape[0] * mc.shape[1]) - white_pixels
        
        # Calculate area
        area = 0

        for c, [_, _, _, Pa] in zip(con, hi[0]):
            if Pa < 0:
                area = area + cv2.contourArea(c)
        
        # Select the roi as the green rectangles
        bndu, bndl = (mc.shape[0]//4, 0), (mc.shape[0]*3//4, mc.shape[1])
        roi = mc[bndu[1]:bndl[1], bndu[0]:bndl[0]]
        
        white_pixels_roi = cv2.countNonZero(roi)
        black_pixels_roi = (roi.shape[0] * roi.shape[1]) - white_pixels_roi
        
        # Few contours and max area to choose the correct image
        # White pixels in the roi are more than the other white ones around
        # Black pixels around are more than the white pixels in the roi
        
        cv2.imshow('Mask {}'.format(n), mc)
        
        if (white_pixels_roi >= (white_pixels - white_pixels_roi) and area > max and len(con) < min
               and (black_pixels - black_pixels_roi) > white_pixels_roi
            or white_pixels_roi >= (white_pixels - white_pixels_roi) and area > max and len(con) >= min
               and len(con) <= mean_contours and (black_pixels - black_pixels_roi) > white_pixels_roi):
            min = len(con)
            max = area
            i_n = n
            imask = mc # Interesting mask
    
    return imask, i_n


# In[3]:


import time

#import maxima
import extract
import contours

img = tables['5']
# img = cv2.imread("ref/Plate16.jpg") #16 7 3 12 4 5 6 2 10

#weight are approximated                        ^^
# img = cv2.bilateralFilter(img, 9, 125, 50)
print("Image of shape {}".format(img.shape))

r, c = img.shape[:2]
t0 = time.time()
ms = extract.get_masks(img, False)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

imask, i_n = select_image(ms)

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

