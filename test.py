import common

import cv2
import numpy as np

import random
import getopt
import sys

import extract
import generator
import knn
import svm
import ocr
import sift

HELP_MESSAGE = """Help:
-h, --help              Displays this help
-k, --ocr <type>        Select the type of ocr
-t, --train             Trains the ocr
-s, --size <int>        Selects the size of the train set [default = 2]
-l, --load <file>       Loads the trained file
-d, --dump <file>       Saves the trained data
-v, --verbose           Verbose prints
--debug                 Enables debug features
-a, --accuracy <int>    Calculates the accuracy
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], "k:tl:d:vs:ha:", ["ocr", "train", "load", "dump", "verbose", "size", "help", "debug", "accuracy"])
except getopt.GetoptError:
    print("Wrong argument")
    print(HELP_MESSAGE)
ocr_types = {
        "knn" : knn.KnnOCR,
        "svm" : svm.SvmOCR,
        "none" : ocr.MockOCR,
        "sift" : sift.SiftOCR
        }

settings = {
        "k" : ocr.MockOCR,
        "t" : False,
        "s" : 2,
        "l" : None,
        "d" : None,
        "v" : False,
        "db" : False,
        "a" : 0
        }
for opt, arg in opts:
    if opt in ("-k", "--ocr"):
        settings["k"] = ocr_types[arg]
    elif opt in ("-t", "--train"):
        settings["t"] = True
    elif opt in ("-l", "--load"):
        settings["l"] = arg
    elif opt in ("-d", "--dump"):
        settings["d"] = arg
    elif opt in ("-v", "--verbose"):
        settings["v"] = True
    elif opt in ("-s", "--size"):
        settings["s"] = int(arg)
    elif opt in ("-h", "--help"):
        print(HELP_MESSAGE)
        sys.exit(1)
    elif opt in ("--debug"):
        settings["db"] = True
    elif opt in ("-a", "--accuracy"):
        settings["a"] = int(arg)

print(settings)
#gen = True
#
#if gen:
#    g = random.choice("1234567890")
#    img = generator.get_all_tables(g)[g]
#    print("Chosen: {}".format(g))
#else:
#    img = cv2.imread("ref/gen/4.jpg")
#
#cv2.imshow("Base image", img)
#
#best_fit, n = extract.get_optimal_mask(img)
#
#cv2.imshow("Best fit: {}".format(n), best_fit)
#
#cl = knn.KnnOCR
#l="data_set"
#
#with cl(load=l) as o:
#    z=o.read(best_fit)
#    print(z)
#
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
