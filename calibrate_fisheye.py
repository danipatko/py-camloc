import struct
import time
import cv2 as cv
import numpy as np
from util import save_remap
from config import *

# Define the chess board rows and columns
CHECKERBOARD = (INNER_COL_LENGTH, INNER_ROW_LENGTH)
subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


def check_chessboard(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

    if not ret:
        return False

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
    imgpoints.append(corners2)

    cv.drawChessboardCorners(frame, (INNER_COL_LENGTH, INNER_ROW_LENGTH), corners2, ret)
    return True


cv.namedWindow('image')
cap = cv.VideoCapture(0)

# NOTE: some devices don't support setting the resolution on the device
# in this case, WIDTH and HEIGHT should be set to the frame's shape
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

found = 0
last_change = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    if found >= TEST_COUNT:
        break

    if cv.waitKey(1) == ord('q'):
        exit(0)

    # show camera
    diff = time.time() - last_change
    if (diff < VID_DELAY):
        cv.putText(frame, "%.2f" % (VID_DELAY - diff), (40, 60), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
        cv.imshow('image', frame)
        continue

    ok = check_chessboard(frame)

    if ok:
        found += 1
        print(f"Found chessboard pattern {found}/{TEST_COUNT}")
        cv.imshow('image', frame)
        cv.waitKey(RES_DELAY * 1000)
    else:
        print("Unable to find pattern")

    last_change = time.time()


K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(TEST_COUNT)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(TEST_COUNT)]

rms, _, _, _, _ = cv.fisheye.calibrate(
    objpoints,
    imgpoints,
    (WIDTH, HEIGHT),
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (WIDTH, HEIGHT), cv.CV_16SC2)
save_remap(SAVE_TO, map1, map2)

cv.destroyAllWindows()
