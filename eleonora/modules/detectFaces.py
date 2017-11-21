"""
This module was made to detect faces.
The module will find, resize, crop and save the detected faces.
"""

import cv2
import time
from eleonora.common.constants import *

counter = 0;
process_this_frame = True

faceCascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)
cap = cv2.VideoCapture(0)


def saveFace(gray, x, y, w, h):
    # Save just the rectangle faces in SubRecFaces
    sub_face = gray[y:y+h, x:x+w]
    FaceFileName = "./eleonora/data/faces/face_" + str(time.time()) + ".jpg"
    cv2.imwrite(FaceFileName, sub_face)

def showFrameRect(gray, x, y, w, h):
    # Shows Rect on frame
    cv2.rectangle(gray, (x, y), (x + w, y + h),(255,255,0), thickness=2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(gray, "Face Detected", (x + 6, y - 6), font, .5, (255, 255, 255), 1)

def countDetection(faces):
    global counter

    if len(faces) > 0:
        if counter >= 20:
            counter = 20
        else:
            counter = counter + 1
    elif len(faces) == 0:
        counter = 0
    elif counter != 0:
        counter = counter - 1
    else:
        counter = 0

def detectFaces():
    global counter

    print ('[' + T + '+' + W + '] Looking for faces')

    while process_this_frame:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        countDetection(faces)

        print("Faces Detect: %s%s%s" % (T, len(faces), W), end="\r")

        for idx, f in enumerate(faces):
            (x, y, w, h) = f

            if counter >= NUMBER_OF_DETECTION:
                showFrameRect(gray, x, y, w, h)

            if counter == NUMBER_OF_DETECTION:
                saveFace(gray, x, y, w, h)

        cv2.imshow('Video', gray)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
