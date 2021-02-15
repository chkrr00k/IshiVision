import numpy as np
import cv2
from matplotlib import pyplot as plt

import maxima
import categorize as cat
import visual

import time #for performace testing

img = cv2.imread("plate5.jpg")

img = cv2.bilateralFilter(img, 9, 150, 75)

print("Started tracking perfomance")
t0 = time.time()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv[:,:,2] = 255

cv2.imshow("Converted", cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))

print("Tracking {} pixels: {}".format(img_hsv.shape[0]*img_hsv.shape[1], img_hsv.shape[:-1]))

img_red = cat.reduce_2_chan(img_hsv)

buck = cat.get_buckets(img_red)

print("Found {} buckets".format(len(buck)))
print("Tracked {} pixels".format(sum([len(buck[p]) for p in buck])))
t1 = time.time()
old_l = len(buck)
buck = cat.filter_nth_percentile_bucket(buck, 40)
print("Percentiled out {} elements".format(old_l - len(buck)))

#if present, gets rid of white
if (0, 0) in buck:
    del buck[(0,0)]

popular = cat.sort_buckets(buck)

#this i useless XXX delete
mx = max(popular, key=lambda e: len(buck[e]))
print("Global maximum: {}".format(mx))

t2 = time.time()
lmx, groups = maxima.local(buck, popular, epsilon=50, distance=maxima.ma_dist_init(maxima.DEF_MALA_MATRIX))
print("Local maxima at: {}".format(lmx))
t3 = time.time()

rgb_map = visual.create_maxima_map(buck, lmx)

LIMIT = (img_hsv.shape[0]*img_hsv.shape[1])/(len(groups)*4)
print("Defined {} as minimum pixel limit".format(LIMIT))

for c in lmx:
    se = visual.create_map(groups[c])

    m, points = visual.mask_from_group(groups[c], buck, img_hsv.shape)

    space = (*c, 255)
    if not len(points) > LIMIT:
        cv2.putText(rgb_map, "x", (c[0]-4, c[1]+4), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        print("Rejected {} due to only {} pixel in its space".format(space, len(points)))
    else:
        cv2.imshow("Map: {}".format(space), m)
        cv2.imshow("{}".format(space), se)
t4 = time.time()
print("""Performance report:
{:.3f}s to categorize in buckets
{:.3f}s to find local maxima
{:.3f}s total
""".format(t1-t0, t3-t2, t4-t0))

cv2.imshow("Local Maxima", rgb_map)
cv2.waitKey(0)


cv2.destroyAllWindows()

