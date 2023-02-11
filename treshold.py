# NOTE: calibrate camera before running

import cv2 as cv
import imutils
from util import load
from config import *

cameraMatrix, distCoeffs, _ = load(SAVE_TO)

cv.namedWindow('image')
cap = cv.VideoCapture(0)

SUM_WEIGHTS = sum(WEIGHTS)
previous = []


def weighted_avg():
    sumX, sumY = (0, 0)
    for i, w in enumerate(WEIGHTS):
        sumX += w * previous[i][0]
        sumY += w * previous[i][1]

    return (sumX // SUM_WEIGHTS, sumY // SUM_WEIGHTS)


while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    frame = cv.undistort(frame, cameraMatrix, distCoeffs, None, None)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, BIN_TRESHOLD, 255, cv.THRESH_BINARY)

    contours = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    sumX = 0
    sumY = 0
    total = 0

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        # filter rects within area limits and check if they are squares
        if MIN_AREA < w * h < MAX_AREA and abs(1 - (w / h)) < SQUARE_RATIO_OFFSET:
            sumX += x
            sumY += y
            total += 1
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if total != 0:
        x = int(sumX / total)
        y = int(sumY / total)

        previous.insert(0, (x, y))
        if len(previous) > len(WEIGHTS):
            previous.pop()

        if len(previous) < len(WEIGHTS):
            continue

        avgX, avgY = weighted_avg()
        if DIST_IGNORE > abs(x - avgX) + abs(y - avgY):
            cv.circle(frame, (avgX, avgY), 5, (0, 0, 255), -1)

    cv.imshow('image', frame)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
