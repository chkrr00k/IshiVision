import common

import cv2
import numpy as np

import ocr
import matcher

class SiftOCR(ocr.OCR):
    def __init__(self, train_set, infos):
        self.sift = cv2.SIFT_create()
        self.train_set = train_set
        self.infos = infos

    def read(self, input, k=2, mmc=4, ip=dict(algorithm=1, trees=5), sp=dict(checks=50), verbose=False):
        input = cv2.medianBlur(input, 11)
        
        kp, des = self.sift.detectAndCompute(input, None)
        flann = cv2.FlannBasedMatcher(ip, sp)
        for (l, m), i in zip(self.train_set.items(), self.infos):
            matches = [m for m, n in flann.knnMatch(des, i.des, k=k) if m.distance < .75 * n.distance]
            if len(matches) > mmc:
                src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([i.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 20.0)
                if M is not None:
                    if verbose:
                        print("Found an homography for {}".format(l))
                    matMask = mask.ravel().tolist()
                    h, w = input.shape
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
                    dst_i = cv2.perspectiveTransform(pts, M)
                    dp = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matMask, flags=2)
            if verbose:
                print("matches('{}') := {}".format(l, len(matches)))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass

    @staticmethod
    def get_train_set():
        c = matcher.get_all_glyphs_refs(ocr.GLYPHS)
        infos = matcher.get_infos(c)
        return c, infos

if __name__ == "__main__":
    import generator
    import extract
    
    ts, i = SiftOCR.get_train_set()
    with SiftOCR(ts, i) as o:
        t, _ = extract.get_optimal_mask(generator.get_all_tables("3")["3"])
        r = o.read(t)
