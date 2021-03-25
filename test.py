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
-c, --char <char>       Specify the char to test
--gkt                   Enables gkt fixes
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], "k:tl:d:vs:ha:c:", ["ocr", "train", "load", "dump", "verbose", "size", "help", "debug", "accuracy", "char", "gkt"])
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
        "a" : 0,
        "c" : None
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
    elif opt in ("-c", "--char"):
        settings["c"] = arg
    elif opt in ("--gkt"):
        print("Not implemented yet")

if settings["l"] and settings["d"]:
    print("Setting -l and -d makes sense only if you want to copy your trainset, be aware!")
if settings["t"] and settings["l"]:
    print("Either -t or -l may be specified")
    sys.exit(-1)
if settings["a"] > 0 and settings["c"] is not None:
    print("Either --accuracy test or -char may be specified")


verbose = settings["v"]
debug = settings["db"]
ocr_class = settings["k"]
train = settings["t"]
load = settings["l"]
dump = settings["d"]
accuracy = settings["a"]
character = settings["c"]
size = settings["s"]

if debug:
    print("Debug mode: ON")
    print("Settings: {}".format(settings))
    common.debug = True

if accuracy > 0:
    hits = 0
    with ocr_class(dump=dump, load=load, verbose=verbose) as o:
        for i in range(accuracy):
            c = str(i%10)
            img, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c])
            r = o.read(img)
            if r == c:
                hits += 1
    print("Accuracy: {:.2f} ({}/{})".format(hits/accuracy, hits, accuracy))
else:
    character = character if character is not None else random.choice("1234567890")
    if debug:
        print("Chosen '{}'".format(character))

    img, _ = extract.get_optimal_mask(generator.get_all_tables(character)[character])
    with ocr_class(dump=dump, load=load, verbose=verbose) as o:
        r = o.read(img)

    print("Found '{}' {}".format(r, "({} was expected)".format(character) if r != character else ""))

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
