import cv2 as cv

TEST_COUNT = 10

cv.namedWindow('image')
cap = cv.VideoCapture(0)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    cv.imshow('image', frame)

    if cv.waitKey(1) == ord('q'):
        cv.imwrite(f"dump{i}.png", frame)
        i += 1
        if i > TEST_COUNT:
            break

cv.destroyAllWindows()
