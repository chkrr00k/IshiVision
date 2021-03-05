import common

import cv2
import numpy as np
import time
import math

import extract
import utils
import maxima
import selector

img = cv2.imread("ref/gen/2.jpg") #16 7 3 12 4 5 6 2 10
#img = cv2.imread("1.jpg") #16 7 3 12 4 5 6 2 10

#weight are approximated                        ^^
img = cv2.bilateralFilter(img, 9, 125, 50)
print("Image of shape {}".format(img.shape))

r, c = img.shape[:2]
t0 = time.time()
ms = extract.get_masks(img)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

stra = False
for n, m in ms.items():
    r, c = m.shape #this is tecnically useless as they were defined above but are kept to be sure

    #straigten the chars (ocr step 1) not needed
    mc = m
    if stra:
        mc = utils.straight(m)
    #selects the intresting part (the center)
#    bndu, bndl = (r//8, c//7), (r*7//8, c*6//7)
#    mc = mc[bndu[0]:bndl[0], bndu[1]:bndl[1]]
    #finds the contourns
    _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cv2.imwrite("ref/{}.jpg".format(n), mc)

    #rgb for visualization
    mc = cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR)
    
    #puts all the rectangles in a list
    rects = list()
    
    #if Pa == -1 then it's a parent node, select only those as they are the main ones 
    for co, [_, _, _, Pa] in zip(con, hi[0]):
        re = [x, y, w, h] = cv2.boundingRect(co)
        if Pa < 0:
            rects.append(re)

    #compress the rectangles that are superimposed to eachothers to have only the main ones
    r = utils.reduce_sections(rects)
    for [x, y, xx, yy] in r:
        cv2.rectangle(mc, (x, y), (xx, yy), (255, 0, 0), 1)
    selector.score(m, con, r, prt=True, name=n)
    
    #actual intresting parts highlighted here
    cv2.rectangle(mc, (mc.shape[0]//5, mc.shape[1]//8), (mc.shape[0]*4//5, mc.shape[1]*7//8), (0, 255, 0), 1)
    cv2.line(mc, ((mc.shape[0]//2),0), (mc.shape[0]//2, mc.shape[1]), (0, 255, 0), 1)
    cv2.imshow("Mask with main contourns for: {}".format(n), mc)



cv2.imshow("Base image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
