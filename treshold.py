import cv2
import numpy as np
import imutils

cv2.namedWindow('image')
cap = cv2.VideoCapture(0)

MIN_AREA = 20
MAX_AREA = 1000
SQUARE_RATIO_OFFSET = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

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
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if total != 0:
        x = int(sumX / total)
        y = int(sumY / total)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
