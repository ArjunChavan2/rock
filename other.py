import cv2
import numpy as np
from math import sqrt

path1 = "data\img3.jpg"
path2 = "annotated\img3_annotated.jpg"
image = cv2.imread(path1)
XSIZE, YSIZE = image.shape[0:2]
MIN_HOLD_PATH = sqrt((YSIZE * XSIZE) / (0.1 * XSIZE))  # Add parentheses for proper calculation
holds = []


class Hold:
    def __init__(self):
        self.points = []

    def add_point(self, x, y):
        self.points.append((x, y))

    def is_similar(self, color):
        avg = sum(color) / 3.0
        return all(abs(color[i] - avg) < 5 for i in range(3))


def create_hold(image, x, y):
    new_hold = Hold()
    new_hold.add_point(x, y)
    create_hold_recursive(image, new_hold, x, y)
    return new_hold


def create_hold_recursive(image, cur_hold, x, y):
    if not cur_hold.is_similar(image[x][y]):
        return
    for point in cur_hold.points:
        if point == (x, y):
            return
    cur_hold.add_point(x, y)
    if 0 < x < XSIZE - 2:
        create_hold_recursive(image, cur_hold, x - 1, y)
        create_hold_recursive(image, cur_hold, x + 1, y)
    if 0 < y < YSIZE - 2:
        create_hold_recursive(image, cur_hold, x, y - 1)
        create_hold_recursive(image, cur_hold, x, y + 1)


if image is None:
    print("\n\n----------\n\nError: Image not loaded successfully\n\n----------\n\n")
else:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for x in range(YSIZE):
        for y in range(XSIZE):
            new_hold = create_hold(rgb, x, y)
            if not any(hold.is_similar(rgb[x][y]) for hold in holds):
                holds.append(new_hold)
        print(len(holds))

print("Done")
print(len(holds))
