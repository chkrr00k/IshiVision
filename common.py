import gi

#this fixes wrong version of Gtk used on debian 10
gi.require_version("Gtk", "2.0")

from functools import wraps
from time import time

def showtime(func):
    """Prints the time of execution of a function"""
    @wraps(func)
    def _time(*args, **kwargs):
        t0 = time()
        try:
            return func(*args, **kwargs)
        finally:
            t1 = time()
            print("{}: {:.3f}s".format(func.__name__, t1-t0))
    return _time
