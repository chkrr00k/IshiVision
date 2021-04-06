import common

import cv2
import numpy as np

import random
import math

import matcher

#special thanks to @franciscouzo

_color = lambda c: (c & 255, (c >> 8) & 255, (c >> 16) & 255)

def _circle(w, h, min_dia, max_dia):
    r = math.floor(random.triangular(min_dia, max_dia, max_dia*.8 + min_dia*.2) / 2)
    a = random.uniform(0, math.pi * 2)
    dfc = random.uniform(0, min(w, h) * .48 - r)
    x = math.floor(w* .5 + math.cos(a) * dfc)
    y = math.floor(h* .5 + math.sin(a) * dfc)
#    print("{} {} {}".format(x, y, dfc))

    return x, y, r

def _overlaps(input, c, bg):
    x, y, r = c
    p_x = [x, x, x, x-r, x+r, x-r*.93, x-r*.93, x+r*.93, x+r*.93]
    p_y = [y, y-r, y+r, y, y, y+r*.93, y-r*.93, y+r*.93, y-r*.93]
    for i, j in zip(p_x, p_y):
        if input[math.floor(i), math.floor(j)] != bg:
            return True
    return False

def _circle_intersects(c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    return (x2-x1)**2 + (y2-y1)**2 < (r2+r1)**2

def _draw_circle(base, base_bg, input, c, c_on, c_off):
    fc = random.choice(c_on if _overlaps(base, c, base_bg) else c_off)
    x, y, r = c
    cv2.circle(input, (math.floor(y), math.floor(x)), math.floor(r), fc, -1)
    # draw the circle
    return input

@common.showtime
def ishihara(base, size, base_bg, bg, n_circles, c_on, c_off):
    """Returns an ishihara table using as base a given image"""
    result = np.full((*size, 3), bg, dtype=np.uint8)

    min_dia, max_dia = (size[0] + size[1]) / 110, (size[1] + size[0]) / 45

    circles = [_circle(size[0], size[1], min_dia, max_dia)]
    circle = circles[0]

    result = _draw_circle(base, base_bg, result, circle, c_on, c_off)

    for i in range(n_circles):
        t = 0
        while any(_circle_intersects(circle, c) for c in circles):
            t += 1
            circle = _circle(size[0], size[1], min_dia, max_dia)
        circles.append(circle)
        _draw_circle(base, base_bg, result, circle, c_on, c_off)

    return result
#c_on=[_color(0xF9BB82), _color(0xEBA170), _color(0xFCCD84)], c_off=[_color(0x9CA594), _color(0xACB4A5), _color(0xBBB964), _color(0xD7DAAA), _color(0xE5D57D), _color(0XD1D6AF)]
C_ON = [_color(0xF9BB82),  _color(0xFCCD84), _color(0xEBA170)]
C_OFF = [_color(0x9CA594), _color(0xACB4A5), _color(0xa3a260),_color(0xBBB964), _color(0xD7DAAA), _color(0xD1D6AF)]
def get_all_tables(glyphs, heigh=200, bezel=40, thic=20, bg=(255,255,255), n_circles=550, c_on=C_ON, c_off=C_OFF, fonts=[cv2.FONT_HERSHEY_SCRIPT_COMPLEX]):
    """Returns a dictionary with ishihara'd glyphs, parameters may be applied"""
    return {k: ishihara(v, v.shape, 0, bg, n_circles, c_on, c_off) for k, v in matcher.get_all_glyphs_refs(glyphs, heigh, bezel, thic, fonts).items()} #FIXME inapropriate use of dict


if __name__ == "__main__":
    import getopt
    import sys
    import re
    HELP_MESSAGE= """Help:
-h, --help                  Displays this help
-g, --glyph <glyph>         Render the given glyph
-bg <comma separated hex>   Uses the given colors for the background
                            (e.g "0x000000, 0xAAAAAA")
-fg <comma separated hex>   Uses the given colors for the foreground
-v, --verbose               Verbose prints
<file name>                 Will be saved as a .PNG regardless
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "vg:h", ["glyph=", "bg=", "fg=","verbose", "help"])
    except getopt.GetoptError:
        print("Wrong argument")

    settings = {
            "g": random.choice("1234567890"),
            "v": False,
            "bg": C_OFF,
            "fg": C_ON
            }
    def get_colors(input):
        if settings["v"]:
            print("Inputed: {}".format(re.split("[,;]\s*", input)))
        return [_color(int(i, 0)) for i in re.split("[,;]\s*", input)]

    for opt, arg in opts:
        if opt in ("-v", "--verbose"):
            settings["v"] = True
        elif opt in ("--fg"):
            settings["fg"] = get_colors(arg)
        elif opt in ("--bg"):
            settings["bg"] = get_colors(arg)
        elif opt in ("-g", "--glyph"):
            settings["g"] = str(arg)
        elif opt in ("-h", "--help"):
            print(HELP_MESSAGE)
            sys.exit(1)
    
    filename = re.split("\.", args[0])[0] if len(args) > 0 else "base"
    if settings["v"]:
        print("Saving on {}.png".format(filename))
        print("Rendering {}".format(settings["g"]))

    base = matcher.render_glyph(settings["g"], 200)
    img = ishihara(base, base.shape, 0, (255,255,255), 500, settings["fg"], settings["bg"])
    cv2.imwrite("{}.png".format(filename), img)

#    for k, v in get_all_tables("1234567890", c_off=[_color(0xBAB8AF)], c_on=[_color(0x4B4B4B), _color(0x747474)]).items():
    
    
#    cv2.imshow("img", img)
#    cv2.imshow("base", base)

 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
