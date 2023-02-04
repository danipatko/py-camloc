import cv2

cv2.namedWindow('image')
firstFrame = None
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("dump.png", frame)
        break

cv2.destroyAllWindows()
