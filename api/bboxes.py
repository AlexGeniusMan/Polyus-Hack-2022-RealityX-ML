Y_CROP = (240, 720)
X_CROP = (200, 900)
SIZE = (X_CROP[1] - X_CROP[0], Y_CROP[1] - Y_CROP[0])

def rescale_bbox(bbox):
    x, y, w, h = bbox
    im_w, im_h = SIZE
    x = int(round((x - w / 2) * im_w + X_CROP[0]))
    y = int(round((y - h / 2) * im_h + Y_CROP[0]))
    w = int(round(w * im_w))
    h = int(round(h * im_h))
    return (x, y, w, h)
