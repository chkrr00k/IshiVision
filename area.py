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
    g = (origin[0]+scale, origin[1]+size[1]//2-scale//2, origin[0]+size[0]-scale, origin[1]+size[1]//2+scale//2)
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
    d = cv2.bitwise_and(img, img, mask=msk)[diay:diayy, int(diaxx-(2**.5*s)//2):int(diax+(2**.5*s)//2)]
    
    m = ((diayy-diay)/(diaxx-diax))
    effx = int(diax-diaxx+(2**.5*s))
    msk = np.zeros((diayy-diay, int(2**.5*s)+1), dtype=np.uint8)

    for i, r in enumerate(d[0:diayy-diay]):
        area = r[int(effx-(2**.5*s)+i/m):int(effx+i/m)]
        msk[i, :len(area)] = area[:msk.shape[1]]
    result.append(msk)

    return result

def print_estimation(a,bu,bl,c,d,eu,el,fu,fl,g,dia):
    print("""
            +-------------------------+
            |           {:3d}%          |
            +-------------------------+
         +-----+                   +-----+
         |{:3d}% |                   |{:3d}% |
         +-----+                   +-----+            /      /
         |{:3d}% |                   |{:3d}% |           /      /
         +-----+                   +-----+          /      /
            +-------------------------+            / {:3d}% /
            |           {:3d}%          |           /      /
            +-------------------------+          /      /
         +-----+                   +-----+      /      /
         |{:3d}% |                   |     |
         +-----+                   |{:3d}% |
         |{:3d}% |                   |     |
         +-----+                   +-----+
            +-------------------------+
            |           {:3d}%          |
            +-------------------------+


            """.format(a,fu,bu,fl,bl,dia,g,eu,c,el,d))

def analyze(img, grid):
    img = img.copy()
    result = list()
    a,e,f=1,4,5
    for i, im in enumerate(extractRois(img, grid)):
        im[im <= 127] = 0
        im[im > 127] = 1
        im[0,0] = 0 #this is an horrible hack
        im[0,1] = 1
        if i in (a,e,f):
            b, w = np.bincount(im.flatten()[:len(im.flatten())//2])
            result.append(int(w/(b+w)*100))
            im = im.ravel()[len(im.flatten())//2:] 
            im[0] = 0 #this is an horrible hack
            im[1] = 1
        b, w = np.bincount(im.flatten())
        result.append(int(w/(b+w)*100))
    return result

def grid(img):
    bimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bimg = cv2.threshold(bimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    _, con, hi = cv2.findContours(bimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, con, -1, (0, 255, 0))
    print(con)
    

    return img


#re = cv2.imread("ref/ssd0.jpg")
#p = points((20, 32), (84, 130), 20)
#p1 = points((111, 35), (80, 140), 20)

#re = cv2.imread("ref/ssd1.jpg")
#p = points((80, 53), (119, 166), 20)

re = cv2.imread("ref/ssd2.jpg")
p = points((95, 50), (105, 170), 20)

#re = cv2.imread("ref/ssd3.jpg")
#p = points((142, 74), (85, 130), 20)

#re_ = drawSSD(re, p)
#re_ = drawSSD(re_, p1)

#cv2.imshow("A", re_)


#re = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)
#print_estimation(*analyze(re, p))

cv2.imshow("R", grid(re))


cv2.waitKey(0)
cv2.destroyAllWindows()
