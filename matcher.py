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
    cv2.putText(result, glyph, center, cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), thic)
    
    return result

def get_all_glyphs_refs(chars, heigh=200, bezel=20, thic=24):
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
#sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

from neighbour import clean

sub = clean(sub)
sub = sub[:,:127]

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(sub, None)

max_v, max_img, max_i = 0, None, "nothing"
FIKT = 1
MMC = 4
ip = dict(algorithm=FIKT, trees=5)
sp = dict(checks=50)


flann = cv2.FlannBasedMatcher(ip, sp)
for (l, m), i in zip(c.items(), infos):
    #cv2.imshow("Base {}".format(l), cv2.drawKeypoints(m, i.kp, m))
    #bf = cv2.BFMatcher()
    
    matches = [m for m, n in flann.knnMatch(des, i.des, k=2) if m.distance < .7 * n.distance]
    if len(matches) > MMC:
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([i.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is not None:
            print("Found an homography for {}".format(l))
            matMask = mask.ravel().tolist()
            h, w = sub.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
            dst_i = cv2.perspectiveTransform(pts, M)
            #m = cv2.polylines(m, [np.int32(dst_i)], True, 255, 3, cv2.LINE_AA)
            dp = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matMask, flags=2)
            img3 = cv2.drawMatches(sub, kp, m, i.kp, matches, None, **dp)
            cv2.imshow("Homo {}".format(l), img3)

    cv2.imshow(l, cv2.drawMatchesKnn(sub, kp, m, i.kp, [[m] for m in matches], None, flags=2))
    print("matches('{}') := {}".format(l, len(matches)))
    if max_v < len(matches):
        max_v = len(matches)
        max_img = m
        max_i = l


#cv2.imshow("Res", sub)
print("Found {} with ({}) matches".format(max_i, max_v))

cv2.waitKey(0)
cv2.destroyAllWindows()
