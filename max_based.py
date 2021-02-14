import numpy as np
import cv2
from matplotlib import pyplot as plt

import maxima

img = cv2.imread("plate3.jpg")

img = cv2.bilateralFilter(img, 9, 75, 75)
#cv2.imshow("Loaded", img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv[:,:,2] = 255
#img_hsv[:,:,1] = 255

cv2.imshow("Converted", cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))

#redmask = cv2.inRange(img_hsv, np.array([0, 60, 255]), np.array([10, 255, 255]))
#greenmask = cv2.inRange(img_hsv, np.array([40, 100, 255]), np.array([80, 255, 255]))

#cv2.imshow("Red mask", redmask)
#cv2.imshow("Green mask", greenmask)

print("Tracking {} pixels: {}".format(img_hsv.shape[0]*img_hsv.shape[1], img_hsv.shape[:-1]))

buck = dict()
for x in range(len(img_hsv)):
    for y in range(len(img_hsv[0])):
        px = tuple(v for v in img_hsv[x, y, :2])
        if px in buck:
            buck[px].append((x, y))
        else:
            buck[px] = [(x, y)]
print("Found {} buckets".format(len(buck)))
print("Tracked {} pixels".format(sum([len(buck[p]) for p in buck])))

def dist(a, b):
    return sum([(int(aa)-int(bb))**2 for aa, bb in zip(a, b)])**.5
#if present, gets rid of white
if (0, 0) in buck:
    del buck[(0,0)]

popular = [*buck]
popular.sort(key=lambda e: len(buck[e]), reverse=True)
#print(popular[:20])
#this i useless XXX delete
mx = max(popular, key=lambda e: len(buck[e]))
print("Global maximum: {}".format(mx))

dif = np.zeros((256, 256), np.uint8)
for k, v in buck.items():
    dif[k[1], k[0]] = 255*len(v)//len(buck[mx])
#cv2.imshow("Color plane", dif)

#for k in popular[:20]:
#    print("{} = {}".format(k, len(buck[(k[0], k[1])])))

test = {k: buck[k] for k in popular[:100]}
#lmx, groups = maxima.local(test, [k for k,_ in test.items()], verbose=True, epsilon=120, distance=maxima.ma_dist_init(maxima.DEF_MALA_MATRIX))
lmx, groups = maxima.local(buck, popular, epsilon=50, distance=maxima.ma_dist_init(maxima.DEF_MALA_MATRIX))
rgb_map = cv2.cvtColor(dif, cv2.COLOR_GRAY2BGR)
print("Local maxima: {}".format(lmx))
#print("Labelling: {}".format(g))
for c in lmx:
    cv2.circle(rgb_map, c, 5, (0, 255, 255), 1)
    #cv2.circle(rgb_map, c, 100, (255, 0, 0), 1)
    #cv2.circle(rgb_map, c, 50, (255, 255, 0), 1)


avg_dst = 50
rec = dict()
for c in lmx:
    #if len(groups[c]) > 0:
    #    inf, sup = maxima.compressed_2d_range(groups[c])
    #else:
    #    inf = sup = c
    #print("Assuming rectangle of [{}, {}] for {}".format(inf, sup, c))
    #rec[c] = (*(int(i) for i in inf), *(int(i) for i in sup))
    space = (*c, 255)
    ##inf = (int(space[0]-avg_dst), int(space[1]-avg_dst), 255)
    ##sup = (int(space[0]+avg_dst), int(space[1]+avg_dst), 255)
    #inf = (*(int(i) for i in inf), 255)
    #sup = (*(int(i) for i in sup), 255)
    #m = cv2.inRange(img_hsv, inf, sup)
    #cv2.imshow("{} - [{},{}]".format(space, inf, sup), m)
    #cv2.rectangle(rgb_map, inf[:-1], sup[:-1], (255, 0, 0), 1)
    se = np.zeros((256, 256), np.uint8)
    for px in groups[c]:
        se[px[1], px[0]] = 255
    #print("Current group: {}".format(groups[c]))
    m = np.zeros(img_hsv.shape)
    points = [i for p in groups[c] for i in buck[p]]
    #print(img_hsv.shape[:-1])
    for p in points:
        m[p[0], p[1]] = 255
    LIMIT = (img_hsv.shape[0]*img_hsv.shape[1])/(len(groups)*4)
    if not len(points) > LIMIT:
        #cv2.putText(se, "REJECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print("Rejected {}".format(space))
    else:
        cv2.imshow("Map: {}".format(space), m)
        cv2.imshow("{}".format(space), se)

print(rec)

cv2.imshow("Local Maxima", rgb_map)
cv2.waitKey(0)


cv2.destroyAllWindows()

