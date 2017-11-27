import sys
import cv2
import time
import dlib
import numpy as np
import threading

import eleonora.common.ui as ui
from eleonora.common.constants import *

import eleonora.modules.speech as speech
import eleonora.modules.uiFace as uiFace

from eleonora.utils.inference import *
from eleonora.utils.input import quit
from eleonora.utils.visualizer import *


number_of_detection = 5
wait_to_sleep = 30

eyeWidth, eyeHeight = 80, 100
process_this_frame = True

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./eleonora/data/haarcascades/shape_predictor_68_face_landmarks.dat')

dx = dy = 0
Eleonora = uiFace.CreateFace("EleonoraFace", 255, eyeWidth, eyeHeight, 2, (0,0,0))

def listenWhileSleep():
    while True:
        speech.listen()

        inputSpeech = input("call me > ")

        if inputSpeech.lower() == 'hi':
            run()
            break

        if quit(value=inputSpeech):
            break

def run():
    global dx, dy
    counter = 0
    preFace = 0
    sleepTimer = wait_to_sleep

    cap = cv2.VideoCapture(0)

    while process_this_frame:
        ret, frame = cap.read()
        frame = convertSequence(frame, (.25, .25), True)
        faces = detector(frame, 1)

        counter = startDetection(counter, faces, number_of_detection)

        needToSleep, sleepTimer = timeToSleep(faces, sleepTimer, wait_to_sleep)

        closingEyes = remap(sleepTimer, 0, wait_to_sleep, eyeHeight, 1)

        if needToSleep:
            Eleonora.sleep(dx, dy)
            Eleonora.show()

            destroy(cap)
            listenWhileSleep()
            break

        if len(faces) <= 0:
            dx, dy = backToZero(dx, dy, uiFace._getScreenSizes(), Eleonora.eyeSpace, eyeWidth, eyeHeight)

        Eleonora.moveEyes(dx, dy, closingEyes)

        # Speak when face is detect
        if preFace != len(faces):
            if len(faces) > 0:
                threading.Thread(target=speech.welcome, args=("Hallo, Ik ken jou nog niet.", "nl")).start()
            preFace = len(faces)

        for (i, face) in enumerate(faces):

            shape = predictor(frame, face)
            shape = shape_to_np(shape)

            (x, y, w, h) = rect_to_bb(face)
            dx, dy = int(x*2), int(y*2)

            if counter >= number_of_detection:
                draw_bounding_box((x, y, w, h), frame, (0, 255, 0))
                cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


        cv2.imshow('Video', frame)

        Eleonora.show()

        if quit(interface=cv2):
            break

    # When everything is done, release the capture
    destroy(cap)


def main(args):
    ui.clearScreen()
    print ('[' + T + '*' + W + '] Starting Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    run()
