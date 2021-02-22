import cv2
import numpy as np

#  aaaa
# f   *b      
# f  * b
#  gggg
# e *  c
# e*   c
#  dddd  
# * = dia
#(x, y), (w, h)
def points(origin, size, scale):
#regions
    a = (*origin, origin[0]+size[0], origin[1]+scale)
    b = (origin[0]+size[0]-scale, origin[1], origin[0]+size[0], origin[1]+size[1]//2)
    c = (origin[0]+size[0]-scale, origin[1]+size[1]//2, origin[0]+size[0], origin[1]+size[1])
    d = (origin[0], origin[1]+size[1]-scale, origin[0]+size[0], origin[1]+size[1])
    e = (origin[0], origin[1]+size[1]//2, origin[0]+scale, origin[1]+size[1])
    f = (origin[0], origin[1], origin[0]+scale, origin[1]+size[1]//2)
    g = (origin[0], origin[1]+size[1]//2-scale//2, origin[0]+size[0], origin[1]+size[1]//2+scale//2)
    dia = (origin[0]+size[0]-scale//2, origin[1]+scale//2, origin[0]+scale//2, origin[1]+size[1]-scale//2, scale)
    
    return dia, a, b, c, d, e, f, g


def drawSSD(img, p):
    img = img.copy()
    ([diax, diay, diaxx, diayy, s], [ax, ay, axx, ayy], [bx, by, bxx, byy], [cx, cy, cxx, cyy], [dx, dy, dxx, dyy], [ex, ey, exx, eyy], [fx, fy, fxx, fyy], [gx, gy, gxx, gyy]) = p
    cv2.rectangle(img, (ax, ay), (axx, ayy), (0,0,255), 1)
    cv2.rectangle(img, (bx, by), (bxx, byy), (0,255,0), 1)
    cv2.rectangle(img, (cx, cy), (cxx, cyy), (255,0,0), 1)
    cv2.rectangle(img, (dx, dy), (dxx, dyy), (127,127,0), 1)
    cv2.rectangle(img, (ex, ey), (exx, eyy), (127,0,127), 1)
    cv2.rectangle(img, (fx, fy), (fxx, fyy), (0,127,127), 1)
    cv2.rectangle(img, (gx, gy), (gxx, gyy), (0,255,255), 1)
    cv2.line(img, (diax-s//2, diay-s//2), (diaxx-s//2, diayy-s//2), (255,255,0), 1)
    cv2.line(img, (diax+s//2, diay+s//2), (diaxx+s//2, diayy+s//2), (255,255,0), 1)
    return img

def extractRois(img, p):
    r = p[1:]
    result = list()

    # abcdefg segments rois
    for [x,y, xx,yy] in r:
        result.append(img[y:yy, x:xx])

    #diagonal roi (straighten)
    msk = np.zeros(img.shape[:2], dtype=np.uint8)
    [diax, diay, diaxx, diayy, s] = p[0]
    cv2.line(msk, (diax, diay), (diaxx, diayy), (255,255,0), s)
    d = cv2.bitwise_and(img, img, mask=msk)[diay:diayy, diaxx:diax]
    result.append(d)

    return result


re = cv2.imread("ref/ssd.jpg")
p = points((20, 32), (84, 130), 25)
p1 = points((111, 35), (80, 140), 25)

re_ = drawSSD(re, p)
re_ = drawSSD(re_, p1)

cv2.imshow("A", re_)

re = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)
for i, im in enumerate(extractRois(re, p1)):
    cv2.imshow("{}".format(i), im)


cv2.waitKey(0)
cv2.destroyAllWindows()
