import numpy as np
import cv2 as cv

INNER_ROW_LENGTH = 13  # 14
INNER_COL_LENGTH = 6

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((INNER_ROW_LENGTH*INNER_COL_LENGTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:INNER_COL_LENGTH, 0:INNER_ROW_LENGTH].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

img = cv.imread("dump.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(
    gray, (INNER_ROW_LENGTH, INNER_COL_LENGTH), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(
        img, (INNER_COL_LENGTH, INNER_ROW_LENGTH), corners2, ret)

    cv.imshow('img', img)
    cv.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    img2 = cv.imread("dump.png")

    # undistort
    h, w = img2.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
    print(newcameramtx)

    dst = cv.undistort(img2, mtx, dist, None, newcameramtx)

    # crop the image
    print(roi)
    x, y, w, h = roi
    if x + y + h + w != 0:
        dst = dst[y:y+h][x:x+w]

    print(dst.shape)

    cv.imshow('img', dst)
    cv.waitKey(0)

cv.destroyAllWindows()
