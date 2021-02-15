import cv2
import numpy as np
import time

import extract

dilate = False
close = True

img = cv2.imread("../ref/Plate16.jpg") #16 7
img = cv2.bilateralFilter(img, 9, 150, 75)
print("Image of shape {}".format(img.shape))

t0 = time.time()
ms = extract.get_masks(img)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

import math
dil_factor = int(math.floor(math.log10(abs(img.shape[0])))) + 5
print("Dilatation factor @ {}".format(dil_factor))

result = list()
for n, m in ms.items():
    dil = m
    if dilate:
        el = cv2.getStructuringElement(cv2.MORPH_CROSS, (int(dil_factor*3), int(dil_factor*3)))
        dil = cv2.dilate(m, el)
    if close:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dil_factor*2), int(dil_factor*2)))
        dil = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)


    _, con, hi = cv2.findContours(~dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c, [_, _, _, Pa] in zip(con, hi[0]):
        if Pa == -1 and (img.shape[0]*img.shape[1])/12 < cv2.contourArea(c):
            [x, y, w, h] = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            roi = dil[y:y+h, x:x+w].copy()

            #inverts the image if it's white dominance
            f = np.bincount(roi.ravel()//255)
            if f[0] < f[1]:
                roi = ~roi
#                cv2.putText(roi, "INV", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            result.append(roi)

    cv2.imshow("Masks dil for {}".format(n), dil)



cv2.imshow("Con", img)
print("Found {} feasible regions".format(len(result)))

for i in range(len(result)):
    cv2.imshow("#{}".format(i), result[i])

cv2.waitKey(0)
cv2.destroyAllWindows()
