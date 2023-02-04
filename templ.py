import cv2
import numpy as np

template = cv2.imread('shid_and_camed.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, (1036 // 10, 2611 // 10))
w, h = template.shape[::-1]

cv2.namedWindow('image')
firstFrame = None
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_gray, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img_gray, kp1, template, kp2,
                           matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('image', img3)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
