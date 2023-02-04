import matplotlib.pyplot as plt
import cv2 as cv

img1 = cv.imread('sample.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('dump.png', cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

offset = 0.005

sumX = 0
sumY = 0

# Apply ratio test
good = []
for m, n in matches:
    if 1 - offset < (m.distance / n.distance) < 1 + offset:
        good.append([m])
        x, y = kp2[m.trainIdx].pt
        sumX += x
        sumY += y

# sorted(matches, key=lambda x: (x[0].distance / x[1].distance))

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good,
                         None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

x = int(sumX // len(good))
y = int(sumY // len(good))
imgtest = cv.circle(img2, (x, y), 10, (0, 255, 0), -1)

cv.imshow('test', imgtest)
plt.imshow(img3), plt.show()
cv.waitKey(0)
