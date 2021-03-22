import common

import cv2
import numpy as np

import matcher

gl = matcher.get_all_glyphs_refs("1234567890")
c = list()
for k, v in gl.items():
    cv2.imshow(k, v)
    print("I:{} m:{}".format(k, cv2.HuMoments(cv2.moments(v))))
    _, cnt, _ = cv2.findContours(v, 2, 1)
    c.append(cnt)

b = cv2.imread("ref/gen/con/4.jpg")
b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
_, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

#print("I:{} m:{}".format("B", cv2.HuMoments(cv2.moments(b))))
cv2.imshow("B", b)
_, cnt, _ = cv2.findContours(b, 2, 1)
#print(cnt)

for i, o in enumerate(c):
    print("{}: {}".format(i, cv2.matchShapes(o[0], cnt[0], cv2.CONTOURS_MATCH_I1, 0)))

cv2.waitKey(0)
cv2.destroyAllWindows()
