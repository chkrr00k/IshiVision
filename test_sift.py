#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import selecter
import matcher
import extract
import generator

c = generator.get_all_tables("1234567890", c_off=[generator._color(0xBAB8AF)], c_on=[generator._color(0x4B4B4B), generator._color(0x747474)])

# print(c)

# c = matcher.get_all_glyphs_refs("1234567890")
print("Calculated {} tables".format(len(c)))

infos = matcher.get_infos(c)


templates = list()

for val, img in c.items():
    ms = extract.get_masks(img)
    imask, _ = selecter.select_image(ms)
    templates.append(imask)

img = cv2.imread("ref/Plate5.jpg") #16 7 3 12 4 5 6 2 10

#weight are approximated                        ^^
img = cv2.bilateralFilter(img, 9, 125, 50)
ms = extract.get_masks(img)
sub, i_n = selecter.select_image(ms)

# sub = cv2.imread("ref/gen/con/4.jpg")
# sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
# _, sub = cv2.threshold(sub, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
# sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

from neighbour import clean

sub = clean(sub)
#sub = sub[:,100:]

sub = cv2.medianBlur(sub, 11)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(sub, None)
cv2.imshow("k", cv2.drawKeypoints(sub.copy(), kp, sub.copy()))

max_v, max_img, max_i = 0, None, "nothing"
FIKT = 1
MMC = 4
ip = dict(algorithm=FIKT, trees=5)
sp = dict(checks=50)


flann = cv2.FlannBasedMatcher(ip, sp)
for (l, m), i in zip(templates.items(), infos):
    #cv2.imshow("Base {}".format(l), cv2.drawKeypoints(m, i.kp, m))
    #bf = cv2.BFMatcher()

    matches = [m for m, n in flann.knnMatch(des, i.des, k=2) if m.distance < .8 * n.distance]
    if len(matches) > MMC:
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([i.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 20.0)
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

