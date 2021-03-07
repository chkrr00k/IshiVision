import common

import cv2
import numpy as np

import extract
import utils
import selector
import visual

img = cv2.imread("ref/gen/1.jpg")
cv2.imshow("Base image", img)

#weight are approximated
img = cv2.bilateralFilter(img, 9, 125, 50)
print("Image of shape {}".format(img.shape))

ms = extract.get_masks(img)

stra = False #straightens the image
write = False #write the masks on disc
show = True #shows the results

scores = list()

for n, m in ms.items():
    #straigten the chars (ocr step 1) not needed
    if stra:
        m = utils.straight(m)

    #finds the contourns
    _, con, hi = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if write:
        cv2.imwrite("ref/{}.jpg".format(n), m)
    
    #puts all the rectangles in a list
    rects, cont = utils.get_parent_contours(con, hi)

    bounds = (m.shape[0]//5, m.shape[1]//8, m.shape[0]*4//5, m.shape[1]*7//8)
    sc = selector.get_score(m, cont, bounds)
    print(selector.get_score_string(*sc, n=n))
    scores.append(sc)
    
    if show:
        #compress the rectangles that are superimposed to eachothers to have only the main ones
        cv2.imshow("{} Mask with main contourns".format(n), visual.print_contours_bounding_rect(m, utils.reduce_sections(rects), bounds))

i = selector.rank(scores)
print("Best fit found: {}".format(i))
n, m = list(ms.items())[i]
if show:
    cv2.imshow("Best fit: {}".format(n), m)




cv2.waitKey(0)
cv2.destroyAllWindows()
