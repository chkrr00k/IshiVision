import cv2
import numpy as np

def render_glyph(glyph, heigh, bezel=20, thic=16):
    scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, heigh)
    size, bl = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_SIMPLEX, scale, thic)
    size = (size[0]+thic+bezel*2, size[1]+bezel*2)
    result = np.zeros(size, dtype=np.uint8)
    center = (0+thic//2+bezel, size[1]-bezel)
    cv2.putText(result, glyph, center, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thic)
    
    return result

def get_all_glyphs_refs(chars, heigh=200, bezel=20, thic=16):
    result = dict()
    for g in chars:
        result[g] = render_glyph(g, heigh, bezel, thic)
    return result

c = get_all_glyphs_refs("1234567890")
for l, m in c.items():
    cv2.imshow(l, m)

cv2.waitKey(0)
cv2.destroyAllWindows()
