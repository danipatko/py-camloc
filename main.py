import cv2
import numpy as np

cv2.namedWindow('image')

threshold = 0.5

firstFrame = None
cap = cv2.VideoCapture(0)

template = cv2.imread('shid_and_camed.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, (1036 // 10, 2611 // 10))
w, h = template.shape[::-1]

template = cv2.Canny(template, 200, 400)

# template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

while True:

    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    asd = cv2.Canny(img_gray, 200, 400)

    res = cv2.matchTemplate(asd, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# cv2.destroyAllWindows()
