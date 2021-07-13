import os
if os.name != "nt":
    import gi

#this fixes wrong version of Gtk used on debian 10
#XXX this has to be removed
    if os.uname()[1] == "melody64":
        gi.require_version("Gtk", "2.0")

from functools import wraps
from time import time

debug = False

def showtime(func):
    """Prints the time of execution of a function"""
    @wraps(func)
    def _time(*args, **kwargs):
        t0 = time()
        try:
            return func(*args, **kwargs)
        finally:
            t1 = time()
            if debug:
                print("{}: {:.3f}s".format(func.__name__, t1-t0))
    return _time
