import cv2 as cv

img1 = cv.imread('sample.jpg', cv.IMREAD_GRAYSCALE)
cap = cv.VideoCapture(0)

# Maximal distance diff between two matches
offset = 0.05

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
# BFMatcher with default params
bf = cv.BFMatcher()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    kp2, des2 = sift.detectAndCompute(frame, None)
    matches = bf.knnMatch(des1, des2, k=2)

    count = 0
    sumX = 0
    sumY = 0

    for m, n in matches:
        # the less diff matches have, the better
        if abs(1 - m.distance / n.distance) < offset:
            count += 1
            x, y = kp2[m.trainIdx].pt
            sumX += x
            sumY += y

    if count == 0:
        continue

    x = int(sumX // count)
    y = int(sumY // count)

    cv.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv.imshow('image', frame)

    if cv.waitKey(1) == ord('q'):
        break
