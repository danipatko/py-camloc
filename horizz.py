import cv2
import numpy as np

gray = cv2.imread("dump.png", cv2.COLOR_BGR2GRAY)

gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY, 15, -2)

horizontal = np.copy(bw)

cols = horizontal.shape[1]
horizontal_size = cols // 30

horizontalStructure = cv2.getStructuringElement(
    cv2.MORPH_RECT, (horizontal_size, 1))

horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

horizontal_inv = cv2.bitwise_not(horizontal)

masked_img = cv2.bitwise_and(gray, gray, mask=horizontal_inv)
masked_img_inv = cv2.bitwise_not(masked_img)

cv2.namedWindow("xdd")
cv2.imshow("xdd", masked_img_inv)
cv2.waitKey(0)
