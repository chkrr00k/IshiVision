import cv2
import numpy as np

def _get_neighbourhood(base, coords, radius=1):
    x, y = coords
    ux, lx = x-radius if x-radius > 0 else 0, x+radius+1 if x+radius+1 < base.shape[0]-1 else base.shape[0]-1
    uy, ly = y-radius if y-radius > 0 else 0, y+radius+1 if y+radius+1 < base.shape[1]-1 else base.shape[1]-1
#    print("[{}:{}, {}:{}]".format(ux, lx, uy ,ly))
    return base[ux:lx, uy:ly]

def clean(input):
    result = np.zeros(input.shape, dtype=np.uint8)

    for c, v in np.ndenumerate(input):
         n = cv2.countNonZero(_get_neighbourhood(input, c, 3).ravel())
         if n > 9:
#             print("{} FG({})".format(c, n))
             result[c] = 255
    return result





i = cv2.imread("ref/ssd3.jpg")
i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
_, i = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

k = clean(i)
cv2.imshow("K", k)
cv2.waitKey(0)
cv2.destroyAllWindows()

#_get_neighbourhood(k, (280,275))

