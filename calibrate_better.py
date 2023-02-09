import cv2 as cv
import time

TEST_COUNT = 10
VID_DELAY = 2  # seconds before taking picture
RES_DELAY = 1  # seconds to show result image for
INNER_ROW_LENGTH = 13
INNER_COL_LENGTH = 6


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def check_chessboard(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (INNER_ROW_LENGTH, INNER_COL_LENGTH), None)

    # Draw the corners
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(frame, (INNER_COL_LENGTH, INNER_ROW_LENGTH), corners2, ret)

    return ret


cv.namedWindow('image')
cap = cv.VideoCapture(0)

i = 0
mode_vid = True
last_change = time.time()
good_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    if (mode_vid and time.time() - last_change > VID_DELAY) or (not mode_vid and time.time() - last_change > RES_DELAY):
        last_change = time.time()
        mode_vid = not mode_vid
        print(f"taking next selfie in {VID_DELAY} seconds..." if mode_vid else "calculating result...")

    if mode_vid:
        cv.imshow('image', frame)
        continue

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
