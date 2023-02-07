import numpy as np
import cv2 as cv

# Before you run:
# Make sure to take some sample images using `calibrate.py`
# Hold a chessboard pattern perpendicular to the camera
# The more pictures you take the better the distortion correction is

INNER_ROW_LENGTH = 13
INNER_COL_LENGTH = 6

TEST_COUNT = 11

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((INNER_ROW_LENGTH*INNER_COL_LENGTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:INNER_COL_LENGTH, 0:INNER_ROW_LENGTH].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

for i in range(0, TEST_COUNT):
    img = cv.imread(f"dump{i}.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(
        gray, (INNER_ROW_LENGTH, INNER_COL_LENGTH), None)

    # If found, add object points, image points (after refining them)
    print(f"{i}: {ret}")
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(
        #    img, (INNER_COL_LENGTH, INNER_ROW_LENGTH), corners2, ret)

        # cv.imshow('img', img)
        # cv.waitKey(0)

# TODO: find a way to store calibration values
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# undistort
cv.namedWindow('image')
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    h, w = frame.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    cv.imshow('image', dst)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
