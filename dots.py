import common
import cv2
import numpy as np

import matcher

import warnings

warnings.warn("The use of the dots module is not suggested by {} for any use".format("chkrr00k"), DeprecationWarning)

def get_grid(step, size, fill=255):
    """Returns the basic stepping dots"""
    result = np.zeros(size, np.uint8)
    result[0::step, 0::step] = fill # [start:stop:step]
    return result

def dot_glyphs(step, chars):
    """Returns the glyphs in dot mode"""
    return {k: cv2.bitwise_and(v, v, mask=get_grid(step, v.shape)) for k, v in matcher.get_all_glyphs_refs(chars).items()}

def dot_clean(input, step, fill=255):
    """Uses the clean algorithm to remove useless dots"""
    def _neigh(b, c, r, s):
        x, y = c
        h = [x+(i*s) for i in range(-r, r+1) if 0 < x+(i*s) < b.shape[0]-1]
        v = [y+(i*s) for i in range(-r, r+1) if 0 < y+(i*s) < b.shape[1]-1]
        #print("{} {}".format(h, v))
        return [b[i, j] for i in h for j in v]

    result = np.zeros(input.shape, dtype=np.uint8)

    for [c] in cv2.findNonZero(get_grid(step, input.shape)):
        n = len(list(filter(lambda e: e > 127, _neigh(input, c, 2, step))))
        if n > 6:
            result[c[0], c[1]] = 255
#            print("{} {} ({})".format(c, _neigh(input, c, 2, step), n))
    return result

if __name__ == "__main__":
    sub = cv2.imread("ref/gen/con/1.jpg")
    sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _, sub = cv2.threshold(sub, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    #sub = sub[10:120, 100:120]

    g = get_grid(6, sub.shape)
    #cv2.imshow("G", g)

    r = cv2.bitwise_and(sub, sub, mask=g)
    #cv2.imshow("M", r)

    g = dot_clean(r, 6)

    cv2.imshow("C", g)
    #rp = np.float32(cv2.findNonZero(g)).reshape(-1,1,2)

    #sh = dot_glyphs(6, "1234567890")

    #for k, v in sh.items():
    #    vp = np.float32(cv2.findNonZero(v)).reshape(-1,1,2)
    ##    print("r{} v{}".format(len(rp), len(vp)))
    ##    cv2.imshow(k, v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
