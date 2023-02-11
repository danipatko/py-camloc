import cv2 as cv
from util import load
from config import SAVE_TO

cameraMatrix, distCoeffs, _ = load(SAVE_TO)

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
