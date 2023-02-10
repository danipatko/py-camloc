import cv2 as cv

cv.namedWindow('image')
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    cv.imshow('image', frame)

    if cv.waitKey(1) == ord('q'):
        cv.imwrite("dump.png", frame)
        break

cv.destroyAllWindows()
