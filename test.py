import cv2
import numpy as np
import time

import extract

dilate = False
close = False

#img = cv2.imread("plate4.jpg") #16 7
img = cv2.imread("../ref/Plate12.jpg") #16 7 3

img = cv2.bilateralFilter(img, 9, 150, 75)
print("Image of shape {}".format(img.shape))

t0 = time.time()
ms = extract.get_masks(img)
t1 = time.time()
print("Masks calculated in: {:.3f}s".format(t1-t0))

import math
dil_factor = int(math.floor(math.log10(abs(img.shape[0])))) + 5
print("Dilatation factor @ {}".format(dil_factor))

def rot(img, ang):
    ic = tuple(np.array(img.shape[1::-1])/2)
    rm = cv2.getRotationMatrix2D(ic, ang, 1.0)
    return cv2.warpAffine(img, rm, img.shape[1::-1], flags=cv2.INTER_LINEAR)

result = list()
for n, m in ms.items():
#    dil = rot(m, 15)
#    e = cv2.Canny(dil, 100, 200)
#    cv2.imshow("Canny 4 {}".format(n), e)
    dil = m

    if dilate:
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (int(dil_factor), int(dil_factor)))
        dil = cv2.dilate(dil, el)
    if close:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dil_factor*2), int(dil_factor*2)))
        dil = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, ker)


    #~dil
    _, con, hi = cv2.findContours(~dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    p = None
    for c, [_, _, _, Pa] in zip(con, hi[0]):

        if Pa == -1 and (img.shape[0]*img.shape[1])/12 < cv2.contourArea(c):
            [x, y, w, h] = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            roi = dil[y:y+h, x:x+w].copy()
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            #inverts the image if it's white dominance
            f = np.bincount(roi.ravel()//255)
            if f[0] < f[1]:
                roi = ~roi
                cv2.putText(roi, "INV", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            result.append(roi)
            p = roi
        elif p is not None:
            [x, y, w, h] = cv2.boundingRect(c)
            cv2.rectangle(p, (x, y), (x+w, y+h), (255, 0, 0), 1)

    cv2.imshow("Masks dil for {}".format(n), dil)



cv2.imshow("Con", img)
print("Found {} feasible regions".format(len(result)))

for i in range(len(result)):
    cv2.imshow("#{}".format(i), result[i])

    #cv2.imshow("C {}".format(i), e)
cv2.waitKey(0)
cv2.destroyAllWindows()
