import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("plate2.jpg")

cv2.imshow("Loaded", img)
whitemask = cv2.inRange(img, np.array([230,230,230]), np.array([255,255,255]))
img = cv2.bitwise_and(img, img, mask=~whitemask)
green = [60, 255, 255]
red = np.array([10, 255, 255])

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
redmask = cv2.inRange(img_hsv, np.array([0, 60, 50]), np.array([10, 255, 255]))
greenmask = cv2.inRange(img_hsv, np.array([20, 100, 100]), np.array([60, 255, 255]))

cv2.imshow("Red Mask", redmask)
cv2.imshow("Green Mask", greenmask)

img_rd = cv2.bitwise_and(img_hsv, img_hsv, mask=~redmask)
img_rd = cv2.bitwise_and(img_rd, img_rd, mask=~greenmask)

cv2.imshow("Disputed", cv2.cvtColor(img_rd, cv2.COLOR_HSV2BGR))

#disputed
buck = dict()
for x in range(len(img_rd)):
    for y in range(len(img_rd[0])):
        px = tuple(v for v in img_rd[x][y])
        if px[0] < 230 and px[1] < 230 and px[2] < 230:
            if px in buck:
                buck[px].append((x, y))
            else:
                buck[px] = [(x, y)]

print("Found {} buckets".format(len(buck)))

def distance(a, b):
    return sum([(aa - bb)**2 for aa, bb in zip(a, b)])**.5

for k, v in buck.items():
    if k[0] > 5 and k[1] > 5 and k[2] > 5:
        rd = distance(k, red)
        gd = distance(k, green)
        if rd > gd:
            for p in v:
                redmask[p[0], p[1]] = 255
        else:
            for p in v:
                greenmask[p[0], p[1]] = 255

cv2.imshow("Amended Red Mask", redmask)
cv2.imshow("Amended Green Mask", greenmask)

cv2.waitKey(0)
cv2.destroyAllWindows()

