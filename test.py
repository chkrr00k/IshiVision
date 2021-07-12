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
-i, --input <file>      Read the number form the given file
-r, --resize <size>     Resize the image before processing. Size must be in 
                        heighxwidth format. Supports the keyword auto for autosizing
-p, --show              Show the images and the internal elaboration passages
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], "k:tl:d:vs:ha:c:j:i:r:p", ["ocr=", "train=", "load=", "dump=", "verbose", "size=", "help", "debug", "accuracy=", "char=", "gtk", "silent", "input=", "resize=", "show"])
except getopt.GetoptError:
    print("Wrong argument")
    print(HELP_MESSAGE)

defaultSize, _ = cv2.getTextSize("8", cv2.FONT_HERSHEY_SIMPLEX, cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, 200), 20)
defaultSize = (defaultSize[0]+20+40*2, defaultSize[1]+40*2)
resize = False

settings = {
        "k": None,
        "t": False,
        "s": 2,
        "l": None,
        "d": None,
        "v": False,
        "db": False,
        "a": 0,
        "c": None,
        "sil": False,
        "j": "modules.json",
        "i": None,
        "r": "x".join((str(d) for d in defaultSize)),
        "p": False
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
    elif opt in ("-i", "--input"):
        settings["i"] = arg
    elif opt in ("-r", "--resize"):
        if arg != "auto":
            settings["r"] = arg
        resize = True
    elif opt in ("-p", "--show"):
        settings["p"] = True

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

if settings["i"] is None and resize:
    print("-i and -r are both asserted. What am i supposed to resize?")
    sys.exit(-8)
if settings["l"] and settings["d"]:
    print("Setting -l and -d makes sense only if you want to copy your trainset, be aware!")
if settings["t"] and settings["l"]:
    print("Both -t and -l may not be specified")
    sys.exit(-1)
if not settings["t"] and settings["l"] is None:
    print("Either -t or -l must be specified")
    sys.exit(-3)
if len(list(filter(lambda a: a, [settings["a"] > 0, settings["c"] is not None, settings["i"] is not None]))) > 1:
    print("Either --accuracy test, --char or --input may be specified")
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
input = settings["i"]
show = settings["p"]



if accuracy > 0:
    hits = 0
    with ocr_class(train_set=train, dump=dump, load=load, verbose=verbose) as o:
        for i in range(accuracy):
            c = str(i%10)
            img, _ = extract.get_optimal_mask(generator.get_all_tables(c)[c], show=show, verbose=verbose)
            r = o.read(img)
            if r == c:
                hits += 1
    if not silent:
        print("Accuracy: {:.2f} ({}/{})".format(hits/accuracy, hits, accuracy))
elif input:
    img = cv2.imread(input)
    if resize:
        size = tuple(int(i) for i in settings["r"].split("x")[::-1])
        if verbose:
            print("Resizing to {}".format(size))
        img = cv2.resize(img, size)
    if img is None or type(img) is not np.ndarray:
        print("Failed to load {} file".format(input))
        sys.exit(-7)

    with ocr_class(train_set=train, dump=dump, load=load, verbose=verbose) as o:
        img, _ = extract.get_optimal_mask(img, show=show, verbose=verbose)
        r = o.read(img)
    if not silent:
        print("Found {}".format(r))
else:
    character = character if character is not None else random.choice("1234567890")
    if debug:
        print("Chosen '{}'".format(character))
    plate = generator.get_all_tables(character)[character]
    if show:
        cv2.imshow("Generated from: '{}'".format(character), plate)
    img, _ = extract.get_optimal_mask(plate, show=show, verbose=verbose)
    with ocr_class(train_set=train, dump=dump, load=load, verbose=verbose) as o:
        r = o.read(img)

    if not silent:
        print("Found '{}' {}".format(r, "({} was expected)".format(character) if r != character else ""))

if show:
    cv2.waitKey()
    cv2.destroyAllWindows()
