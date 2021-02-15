import cv2
import numpy as np
import time

import extract

img = cv2.imread("plate5.jpg")
img = cv2.bilateralFilter(img, 9, 150, 75)

t0 = time.time()
ms = extract.get_masks(img)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

#for n, m in ms.items():
#    #bl = cv2.GaussianBlur(m, (5, 5), 0)
#    #ret, th = cv2.threshold(m, 127, 255, 0)
#    _, con, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#    for c in con:
#        if cv2.contourArea(c) > 50:
#            [x, y, w, h] = cv2.boundingRect(c)
#            if h > 100 and h > w:
#                cv2.rectangle(m, (x, y), (x+w, y+h), (255, 255, 255), 1)
#    
#    cv2.imshow("Masks for {}".format(n), m)
#cv2.imshow("Con", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
