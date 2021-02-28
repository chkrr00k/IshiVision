import common

import cv2
import numpy as np
from collections import namedtuple

import matplotlib.pyplot as plt

def render_glyph(glyph, heigh, bezel=20, thic=26):
    scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, heigh)
    size, bl = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_SIMPLEX, scale, thic)
    size = (size[0]+thic+bezel*2, size[1]+bezel*2)
    result = np.zeros(size, dtype=np.uint8)
    center = (0+thic//2+bezel, size[1]-bezel)
    cv2.putText(result, glyph, center, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thic)
    
    return result

def get_all_glyphs_refs(chars, heigh=200, bezel=20, thic=26):
    result = dict()
    for g in chars:
        result[g] = render_glyph(g, heigh, bezel, thic)
    return result

def get_infos(inputs):
    result = list()
    ImageInfo = namedtuple("ImageInfo", "glyph kp des")

    for l, m in inputs.items():
        sift = cv2.SIFT_create() #XXX only one is needed
        kp, des = sift.detectAndCompute(m, None)
        result.append(ImageInfo(l, kp, des))
    return result



c = get_all_glyphs_refs("1234567890")
print("Calculated {} tables".format(len(c)))

infos = get_infos(c)

sub = cv2.imread("ref/ssd3.jpg")
sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
_, sub = cv2.threshold(sub, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

from neighbour import clean

sub = clean(sub)
sub = sub[:,127:]

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(sub, None)

max_v, max_img = 0, None
for (l, m), i in zip(c.items(), infos):
    #cv2.imshow(l, cv2.drawKeypoints(m, i.kp, m))
    bf = cv2.BFMatcher()
    matches = [[m] for m, n in bf.knnMatch(des, i.des, k=2) if m.distance < .95 * n.distance]
    cv2.imshow(l, cv2.drawMatchesKnn(sub, kp, m, i.kp, matches, None, flags=2))
    print("matches('{}') := {}".format(l, len(matches)))
    if max_v < len(matches):
        max_v = len(matches)
        max_img = m


cv2.imshow("Res", max_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
