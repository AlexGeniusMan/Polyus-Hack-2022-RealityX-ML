Y_CROP = (240, 720)
X_CROP = (200, 900)
SIZE = (X_CROP[1] - X_CROP[0], Y_CROP[1] - Y_CROP[0])

REAL_LEN = 1600

Y_MID_POINT = 440
Y_HIGH_POINT = 340
Y_LOW_POINT = 720
LINE_LOW = (970, 55)
LINE_MID = (800, 250)
LINE_HIGH = (720, 350)
LOWER_RATIO = ((LINE_LOW[0]-LINE_LOW[1])-(LINE_MID[0]-LINE_MID[1])) / (Y_LOW_POINT-Y_MID_POINT)
HIGHER_RATIO = ((LINE_MID[0]-LINE_MID[1])-(LINE_HIGH[0]-LINE_HIGH[1])) / (Y_MID_POINT-Y_HIGH_POINT)

TOLERANCE = 50


def rescale_bbox(bbox):
    x, y, w, h = bbox
    im_w, im_h = SIZE
    x = int(round((x - w / 2) * im_w + X_CROP[0]))
    y = int(round((y - h / 2) * im_h + Y_CROP[0]))
    w = int(round(w * im_w))
    h = int(round(h * im_h))
    return (x, y, w, h)


def bbox_to_mm_max(bbox):
    x, y, w, h = bbox
    y_lower, y_higher = y+h, y
    y_center = y + h/2
    if y_center > Y_MID_POINT:
        y_rast = Y_LOW_POINT - y_center
        local_belt_w = (LINE_LOW[0]-LINE_LOW[1]) - LOWER_RATIO * y_rast
        new_w = w / local_belt_w * REAL_LEN
        new_h = h / w * new_w * LOWER_RATIO
    else:
        y_rast = Y_MID_POINT - y_center
        local_belt_w = (LINE_MID[0]-LINE_MID[1]) - HIGHER_RATIO * y_rast
        new_w = w / local_belt_w * REAL_LEN
        new_h = h / w * new_w * HIGHER_RATIO
    return max(new_w, new_h)


def fuse(intersections: list):
    fused = [intersections[0]]
    intersections = intersections[1:]
    for intersec in intersections:
        if fused[-1][1] >= intersec[0]:
            fused[-1] = (min(fused[-1][0], intersec[0]), max(fused[-1][1], intersec[1]))
        else:
            fused.append(intersec)
    return fused

def get_fullness(intersection_points):
    if len(intersection_points) > 1:
        intersection_points.sort(key=lambda x: x[0])
        intersection_points = fuse(intersection_points)
        fullness = sum([x[1] - x[0] for x in intersection_points]) / (LINE_MID[0] - LINE_MID[1] - TOLERANCE * 2)
        return fullness
    elif len(intersection_points) == 1:
        fullness = sum([x[1] - x[0] for x in intersection_points]) / (LINE_MID[0] - LINE_MID[1] - TOLERANCE * 2)
        return fullness
    else:
        return 0

