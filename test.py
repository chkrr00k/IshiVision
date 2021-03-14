import common

import cv2
import numpy as np

import random

import extract
import generator

gen = True
stra = False #straightens the image
write = False #write the masks on disc
show = True #shows the results

if gen:
    g = random.choice("1234567890")
    img = generator.get_all_tables(g)[g]
    print("Chosen: {}".format(g))
else:
    img = cv2.imread("ref/gen/4.jpg")

cv2.imshow("Base image", img)

best_fit, n = extract.get_optimal_mask(img)

if show:
    cv2.imshow("Best fit: {}".format(n), best_fit)




cv2.waitKey(0)
cv2.destroyAllWindows()
