import numpy as np
import imutils
import dlib
import cv2

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return(x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

number_of_detection = 5
process_this_frame = True


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../eleonora/data/haarcascades/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

counter = 0;

while process_this_frame:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if len(rects) > 0:
        if counter >= 20:
            counter = 20
        else:
            counter = counter + 1
    elif len(rects) == 0:
        counter = 0
    elif counter != 0:
        counter = counter - 1
    else:
        counter = 0


    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)

        if counter >= number_of_detection:
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('Video', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
