#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from util import load

cameraMatrix, distCoeffs, _ = load("sjcam")

cv.namedWindow('image')
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    dst = cv.undistort(frame, cameraMatrix, distCoeffs, None, None)
    cv.imshow('image', dst)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
