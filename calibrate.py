import cv2

TEST_COUNT = 10

cv2.namedWindow('image')
cap = cv2.VideoCapture(0)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite(f"dump{i}.png", frame)
        i += 1
        if i > TEST_COUNT:
            break

cv2.destroyAllWindows()
