import cv2
import numpy as np
import imutils

cv2.namedWindow('image')
cap = cv2.VideoCapture(0)

# adjust accoording to lights (the higher, the less white areas)
BIN_TRESHOLD = 0.8 * 255
# minimal rectangle area
MIN_AREA = 20
# maximal rectangle area
MAX_AREA = 1000
# maximal difference of the rectangle's side ratio
SQUARE_RATIO_OFFSET = 0.1
# drop a frame if the average and the detected point is too far away from each other
DIST_IGNORE = 100

# first elem is the latest frame
WEIGHTS = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
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

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, BIN_TRESHOLD, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(image.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    sumX = 0
    sumY = 0
    total = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # filter rects within area limits and check if they are squares
        if MIN_AREA < w * h < MAX_AREA and abs(1 - (w / h)) < SQUARE_RATIO_OFFSET:
            sumX += x
            sumY += y
            total += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
            cv2.circle(frame, (avgX, avgY), 5, (0, 0, 255), -1)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
