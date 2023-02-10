import cv2 as cv
import time
import numpy as np

SAVE = True
TEST_COUNT = 20
VID_DELAY = 2  # seconds before taking picture
RES_DELAY = 0.5  # seconds to show result image for
INNER_ROW_LENGTH = 13
INNER_COL_LENGTH = 6
WIDTH = 640
HEIGHT = 480

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []
objp = np.zeros((INNER_ROW_LENGTH*INNER_COL_LENGTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:INNER_COL_LENGTH, 0:INNER_ROW_LENGTH].T.reshape(-1, 2)


def check_chessboard(frame):
    copy = frame.copy()
    gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (INNER_ROW_LENGTH, INNER_COL_LENGTH), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(copy, (INNER_COL_LENGTH, INNER_ROW_LENGTH), corners2, ret)
        return copy

    return None


cv.namedWindow('image')
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

i = 0
last_change = time.time()
good_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    if cv.waitKey(1) == ord('q'):
        exit(0)

    # show camera
    if (time.time() - last_change < VID_DELAY):
        cv.imshow('image', frame)
        continue

    last_change = time.time()
    dislpay_frame = check_chessboard(frame)

    if dislpay_frame is not None:
        print("Found chessboard pattern")
        cv.imshow('image', dislpay_frame)

        if SAVE:
            cv.imwrite(f"dump{i}.png", frame)
            i += 1

        time.sleep(RES_DELAY)
    else:
        print("Unable to find pattern")


# evaluate
_, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frame.shape[::-1], None, None)
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (WIDTH, HEIGHT), 1)


# save to file
def save(cameraMatrix, distCoeffs, newCameraMatrix):
    file = open(f"save.cfg", "w")

    res = np.array2string(cameraMatrix, formatter={'float_kind': lambda x: "%.10f" % x}) + "\n"
    res += f"{distCoeffs[0]}{distCoeffs[1]}{distCoeffs[2]}{distCoeffs[3]}\n"
    res += np.array2string(newCameraMatrix, formatter={'float_kind': lambda x: "%.10f" % x}) + "\n"

    file.write(res)
    file.close()


cv.destroyAllWindows()
