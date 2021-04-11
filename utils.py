import cv2
import numpy as np
import math
from functools import reduce


def get_parent_contours(con, hi):
    """Given a contours list and its hierarchy returns the parent contours and their bounding rects"""
    if len(hi) == 0:
        raise ValueError("The hierarchy must have elements in it")
    rects, cont = list(), list()
    for co, [_, _, _, Pa] in zip(con, hi[0]):
        re = cv2.boundingRect(co)
        #if Pa == -1 then it's a parent node, select only those as they are the main ones 
        if Pa < 0:
            rects.append(re)
            cont.append(co)
    return rects, cont


def straight(img, angle=.15):
    """Straigten the text by wraping it of .15 rad on the left"""
    r, c = img.shape[:2]
    shift = math.tan(angle)*r/2
    src, dst = np.float32([[0, 0],[c, 0],[0, r]]), np.float32([[-shift,0],[c-shift, 0],[shift, r]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (r, c), flags=cv2.INTER_LINEAR)

def rot(img, ang):
    """Rotates the matrix of ang degrees"""
    ic = tuple(np.array(img.shape[1::-1])/2)
    rm = cv2.getRotationMatrix2D(ic, ang, 1.0)
    return cv2.warpAffine(img, rm, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def rect_center(r1, convert=False):
    """Returns the center (as defined by centroid) of a rectangle"""
    if convert:
        r1 = rect_convert(r1)
    return (r1[0]+(r1[2]-r1[0])//2, r1[1]+(r1[3]-r1[1])//2)

def rect_in_rect(rs, rb, epsilon=0, convert=False):
    """Tells if rs is totally in rb"""
    if convert:
        rs = rect_convert(rs)
        rb = rect_convert(rb)
    return (rb[0]-epsilon <= rs[0] and rb[1]-epsilon <= rs[1]) and (rs[2] <= rb[2]+epsilon+epsilon and rs[3] <= rb[3])

def point_in_rect(p, r1, convert=False):
    """Returns if a point p is included in the rectangle r1, convert is to convert as define in conversion functions"""
    if convert:
        r1 = rect_convert(r1)
    return r1[0] < p[0] < r1[2] and r1[1] < p[1] < r1[3]

def rect_intersect(r1, r2, convert=False):
    """tells if two rectangles intersects with eachothers. If convert is true conver the coords in (top left, bottom right) format"""
    if convert:
        r1 = rect_convert(r1)
        r2 = rect_convert(r2)
    return not (r1[0] > r2[2] or r1[2] < r2[0] or r1[1] > r2[3] or r1[3] < r2[1])

def rect_fuse(r1, r2, convert=False):
    """Fuse two rectangles. If convert is true convert the coords in (top left, bottom right) format"""
    if convert:
        r1 = rect_convert(r1)
        r2 = rect_convert(r2)
    return (min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3]))

def rect_convert(input):
    """Converts a rectangle from (top left, size) to (top left, bottom right)"""
    return (input[0], input[1], input[0]+input[2], input[1]+input[3])

def rect_deconvert(input):
    """Deconverts a rectangle from (top left, bottom right) to (top left, size)"""
    return (input[0], input[1], input[2]-input[0], input[3]-input[1])

def reduce_sections_total(rois, convert=True):
    """Maximum compress the rectangles in a section"""
    if convert:
        rois = [rect_convert(r) for r in rois]
    i = 0
    while i < len(rois):
        j = 0
        while j < len(rois):
            if i != j and rect_intersect(rois[i], rois[j]):
                n = rect_fuse(rois[i], rois[j])
                rois[i] = n
                print("Fused {}&{}=>{}".format(rois[i], rois[j], n))
                rois.pop(j)
                if i > j:
                    i -= 1
            else:
                j += 1
        i += 1
    return rois
def reduce_sections(rois, convert=True):
    """Reduce the rectangles in an area in only one if the rectangles collide with eachothers. If convert is true it converts the rectangle in the rois in (top left, bottom right) format"""
    if convert:
        rois = [rect_convert(r) for r in rois]
    result = [[rois[0]]]
    for c in rois[1:]:
        for r in result:
            if any(rect_intersect(c, rr) for rr in r):
                r.append(c)
                break
        else:
            result.append([c])
    return [reduce(rect_fuse, r) for r in result]

def reduce_sections_area(cont, convert=True):
    """Reduce the contourns in an area in only one if the bounding rectangles collide with eachothers. If convert is true it converts the rectangle in the rois in (top left, bottom right) format
    Returns a struct with bound and area field for the proper extraction of the results
    """

    rois = [cv2.boundingRect(c) for c in cont]
    if convert:
        rois = [rect_convert(r) for r in rois]
    
    def init(s, a, c):
        s.area = a
        s._children = [c]
    Section = type("Section", (), {"bound":None, "__init__": init})
    
    result = [Section(cv2.contourArea(cont[0]), rois[0])]
    for c, b in zip(cont[1:], rois[1:]):
        for r in result:
            if any(rect_intersect(b, rr) for rr in r._children):
                r._children.append(b)
                r.area += cv2.contourArea(c)
                break
        else:
            result.append(Section(cv2.contourArea(c), b))
    for r in result:
        r.bound = reduce(rect_fuse, r._children)
    return result

assert reduce_sections([[0,0, 1,1], [1,1, 2,2], [4,4, 5,5], [0,0,2,4], [2,2,3,3]]) == [(0,0,5,5),(4,4,9,9)], "Idk"

a = [0,0, 4,4]
b = [2,2, 4,4]
assert rect_in_rect(b, a), "{} inside {} ".format(b, a)
assert rect_intersect(b, a), "{} and {} intersects".format(b, a)

a = [0,0, 4,4]
b = [1,1, 2,2]
c = [3,3, 5,5]
assert rect_intersect(a, b), "{} and {} intersects".format(a, b)
assert rect_intersect(a, c), "{} and {} intersects".format(a, c)
assert rect_intersect(b, a), "{} and {} intersects".format(b, a)
assert rect_intersect(c, a), "{} and {} intersects".format(c, a)

