import common

import cv2
import numpy as np

import extract
import utils
import selector

img = cv2.imread("ref/gen/1.jpg") #16 7 3 12 4 5 6 2 10
#img = cv2.imread("1.jpg") #16 7 3 12 4 5 6 2 10

cv2.imshow("Base image", img)
#weight are approximated                        ^^
img = cv2.bilateralFilter(img, 9, 125, 50)
print("Image of shape {}".format(img.shape))

ms = extract.get_masks(img)

stra = False

scores = list()

for n, m in ms.items():
    #straigten the chars (ocr step 1) not needed
    mc = m
    if stra:
        mc = utils.straight(m)
    #finds the contourns
    _, con, hi = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cv2.imwrite("ref/{}.jpg".format(n), mc)

    #rgb for visualization
    mc = cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR)
    
    #puts all the rectangles in a list

    rects, cont = utils.get_parent_contours(con, hi)

    #compress the rectangles that are superimposed to eachothers to have only the main ones
    r = utils.reduce_sections(rects)
    for [x, y, xx, yy] in r:
        cv2.rectangle(mc, (x, y), (xx, yy), (255, 0, 0), 1)


    bounds = (m.shape[0]//5, m.shape[1]//8, m.shape[0]*4//5, m.shape[1]*7//8)
    sc = selector.get_score(m, cont, bounds)
    print(selector.get_score_string(*sc, n=n))
    scores.append(sc)
    
    #actual intresting parts highlighted here
    cv2.rectangle(mc, (mc.shape[0]//5, mc.shape[1]//8), (mc.shape[0]*4//5, mc.shape[1]*7//8), (0, 255, 0), 1)
    cv2.line(mc, ((mc.shape[0]//2),0), (mc.shape[0]//2, mc.shape[1]), (0, 255, 0), 1)
    cv2.imshow("{} Mask with main contourns".format(n), mc)

i = selector.rank(scores)
print("Best fit found: {}".format(i))
n, m = list(ms.items())[i]
cv2.imshow("Best fit: {}".format(n), m)




cv2.waitKey(0)
cv2.destroyAllWindows()
