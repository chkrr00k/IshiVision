import common

import cv2
import numpy as np

import random
import getopt
import sys
import importlib
import json

import extract
import generator
#import ocr

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
--gkt                   Enables gkt fixes for debian 10 and OpenCV 3.something
--silent                Produce no output
-j <json file>          Select ocr modules file
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], "k:tl:d:vs:ha:c:j:", ["ocr", "train", "load", "dump", "verbose", "size", "help", "debug", "accuracy", "char", "gtk", "silent"])
except getopt.GetoptError:
    print("Wrong argument")
    print(HELP_MESSAGE)

settings = {
        "k" : None,
        "t" : False,
        "s" : 2,
        "l" : None,
        "d" : None,
        "v" : False,
        "db" : False,
        "a" : 0,
        "c" : None,
        "sil" : False,
        "j": "modules.json"
        }
for opt, arg in opts:
    if opt in ("-k", "--ocr"):
        settings["k"] = arg
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
    elif opt in ("--gtk"):
        import gi
        gi.require_version("Gtk", "2.0")
    elif opt in ("--silent") :
        settings["sil"] = True
    elif opt in ("-j"):
        settings["j"] = arg

debug = settings["db"]
verbose = settings["v"]
if debug:
    print("Debug mode: ON")
    print("Settings: {}".format(settings))
    common.debug = True
    verbose = True
try:
    with open(settings["j"], "r") as f:
        modules = json.load(f)
    if settings["k"] in modules["ocr"]:
        settings["k"] = modules["ocr"][settings["k"]]
    elif settings["k"] is None:
        settings["k"] = modules["ocr"][modules["default"]]
    else:
        print("Your selected OCR method (-k) does not exist")
        sys.exit(-2)
except Exception as e:
    print("Your module file is likely misformatted or you selected a non existing OCR method")
    if debug:
        print(e)
    sys.exit(-4)

if settings["l"] and settings["d"]:
    print("Setting -l and -d makes sense only if you want to copy your trainset, be aware!")
if settings["t"] and settings["l"]:
    print("Both -t and -l may not be specified")
    sys.exit(-1)
if not settings["t"] and settings["l"] is None:
    print("Either -t or -l must be specified")
    sys.exit(-3)
if settings["a"] > 0 and settings["c"] is not None:
    print("Either --accuracy test or -char may be specified")
    sys.exit(-5)

try:
    m = importlib.import_module(settings["k"]["mdl"]) # dynamic module import
    ocr_class = getattr(m, settings["k"]["cls"]) # given the module m, get the requested class
except Exception as e:
    print("Failed to load the selected OCR due to \"{}\"".format(e))
    sys.exit(-6)
if debug:
    print("Loaded: {}".format(ocr_class))
load = settings["l"]
dump = settings["d"]
accuracy = settings["a"]
character = settings["c"]
size = settings["s"]
train = size if settings["t"] else None
silent = settings["sil"]



if accuracy > 0:
    hits = 0
    with ocr_class(train_set=train, dump=dump, load=load, verbose=verbose) as o:
        for i in range(accuracy):
            c = str(i%10)
            img, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c])
            r = o.read(img)
            if r == c:
                hits += 1
    if not silent:
        print("Accuracy: {:.2f} ({}/{})".format(hits/accuracy, hits, accuracy))
else:
    character = character if character is not None else random.choice("1234567890")
    if debug:
        print("Chosen '{}'".format(character))

    img, _ = extract.get_optimal_mask(generator.get_all_tables(character)[character])
    with ocr_class(train_set=train, dump=dump, load=load, verbose=verbose) as o:
        r = o.read(img)

    if not silent:
        print("Found '{}' {}".format(r, "({} was expected)".format(character) if r != character else ""))

