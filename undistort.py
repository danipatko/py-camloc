import cv2 as cv
from util import load_remap
from config import SAVE_TO
import numpy as np

map1, map2 = load_remap(SAVE_TO)

cv.namedWindow('image')
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    dst = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    cv.imshow('image', dst)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
