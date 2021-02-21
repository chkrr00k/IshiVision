import cv2
import numpy as np
import time
import math

import extract
import utils
import maxima

#dilate = False
#close = False

#img = cv2.imread("plate4.jpg") #16 7
img = cv2.imread("ref/Plate3.jpg") #16 7 3 12 4 5 6 2 10
#wight are approximated                        ^^
img = cv2.bilateralFilter(img, 9, 125, 50)
print("Image of shape {}".format(img.shape))

r, c = img.shape[:2]
t0 = time.time()
ms = extract.get_masks(img)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

#dil_factor = int(math.floor(math.log10(abs(img.shape[0])))) + 5
#print("Dilatation factor @ {}".format(dil_factor))

for n, m in ms.items():
    r, c = m.shape
    #
    mc = utils.straight(m)
    bndu, bndl = (r//8, c//7), (r*7//8, c*6//7)
    mc = mc[bndu[0]:bndl[0], bndu[1]:bndl[1]]
    _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mc = cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR)
    
    rects = list()
#    area = cv2.contourArea(c)
#    cv2.putText(roi, "Ar: {}".format(area), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for co, [_, _, _, Pa] in zip(con, hi[0]):
        re = [x, y, w, h] = cv2.boundingRect(co)
        if Pa < 0:
#            cv2.rectangle(mc, (x, y), (x+w, y+h), (0, 0, 255), 1)
            rects.append(re)
    for [x, y, xx, yy] in utils.reduce_sections(rects):
        cv2.rectangle(mc, (x, y), (xx, yy), (255, 0, 0), 1)

    cv2.rectangle(mc, (mc.shape[0]//4, 0), (mc.shape[0]*3//4, mc.shape[1]), (0, 255, 0), 1)
    cv2.line(mc, ((mc.shape[0]//2),0), (mc.shape[0]//2, mc.shape[1]), (0, 255, 0), 1)
    cv2.imshow("c:{}".format(n), mc)



cv2.imshow("Con", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
