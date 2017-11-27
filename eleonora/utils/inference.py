import cv2
import time
import math
import numpy as np

def remap(value, low1, high1, low2, high2):
    return low2 + (high2 - low2) * (value - low1) / (high1 - low1);

def roundup(x, step):
    return int(math.ceil(x / step)) * step

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return(x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def convertSequence(f, sizes = (0.25, 0.25), grayscale=True):
    f = cv2.flip(f, 1)
    f = cv2.resize(f, (0, 0), fx=sizes[0], fy=sizes[1])
    if grayscale:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f

def destroy(cap):
    cap.release()

def startDetection(counter, arr, numDetections):
    if len(arr) > 0:
        if counter >= numDetections:
            return numDetections
        else:
            return counter + 1
    elif len(arr) == 0:
        return 0
    elif counter != 0:
        return counter - 1
    else:
        return 0

def timeToSleep(arr, t, tmax):
    if len(arr) <= 0:
        t = t - 1
        if t <= 0:
            return True, t
        else:
            return False, t
    else:
        t = tmax
        return False, t

def backToZero(dx, dy, s, eyeSpace, eyeWidth, eyeHeight):
    steps = 30

    dx, dy = roundup(dx, steps), roundup(dy, steps)
    mx, my = roundup(((s[0] / 2) - (eyeSpace + eyeWidth*1.5)), steps), roundup(eyeHeight, steps)

    if dx > mx:
        dx = dx - steps
    elif dx < mx:
        dx = dx + steps

    if dy > my:
        dy = dy - steps
    elif dy < my:
        dy = dy + steps

    return dx, dy
