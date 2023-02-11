import time
import cv2 as cv
import numpy as np
from util import save
from config import *

# distortion params
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []
objp = np.zeros((INNER_ROW_LENGTH*INNER_COL_LENGTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:INNER_COL_LENGTH, 0:INNER_ROW_LENGTH].T.reshape(-1, 2)


def check_chessboard(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (INNER_ROW_LENGTH, INNER_COL_LENGTH), None)

    if not ret:
        return False

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
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


# evaluate
_, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (WIDTH, HEIGHT), None, None)
newCameraMatrix, _ = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, RESCALE if RESCALE is not None else (WIDTH, HEIGHT), 1)

# save
save(SAVE_TO, cameraMatrix, distCoeffs[0], newCameraMatrix)

cv.destroyAllWindows()
