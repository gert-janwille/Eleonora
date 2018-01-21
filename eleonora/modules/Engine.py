import cv2
import time
import threading
import numpy as np
import eleonora.utils.config as config
from keras.models import load_model, model_from_json

from eleonora.modules import AI, DB, UI
from eleonora.utils.util import *
from eleonora.utils.input import quit, message, header

active_mode = False

def engine():
    global speech, listener, result

    message('Starting Eleonora...')

    # loading models
    emotion_classifier = load_model(config.emotion_model_path, compile=False)
    facial_classifier = model_from_json(getJson(config.facial_json_path_mat))

    # load weights
    AI.load_weights(facial_classifier, config.facial_math_path)

    # getting input shapes
    emotion_target_size = emotion_classifier.input_shape[1:3]
    facial_target_size = facial_classifier.input_shape[1:3]

    # Initialize Database
    db = DB.createDatabase(config.database_path)
    # Initialize HotKeyListener
    listener = AI.HotKeyListener()
    listener.listen(start_Listening)
    # Initialize SpeechRecognition
    speech = AI.SpeechRecognition(lang='nl')
    # speech.welcome()

    # Read Database
    face_db = db.read(key="persons")

    cap = cv2.VideoCapture(0)
    samePerson = (False, 0)

    message('Starting Life...')
    # print(threading.enumerate())

    while config.process_this_frame:
        # Running your own FPS
        if not int(time.time()) % config.MOMENTUM:

            # Read cap and preprocess frame
            frame = cap.read()[1]
            frame, out = convertSequence(frame, grayscale=False)

            # Detect Faces, same person and output file
            flag, samePerson, frame, sFile = detect_faces(frame, emotion_target_size, samePerson[1])

            # if config.VERBOSE:
            #     print(config.scaned_person)

            # Reset Person
            if not flag:
                resetScanedPerson()

            # Reset Time
            if flag and config.scaned_person:
                config.reset_time = 15

            # If no scaned person detect one
            if flag and not config.scaned_person and not config.active_mode:
                message('Face Detected, Ready to identify')

                # Pause listener
                listener.pause()

                # Read Database
                face_db = db.read(key="persons")

                if not samePerson[0]:
                    if db.isEmpty("persons"):
                        person = False
                    else:
                        # Loop over db
                        person = identifyPerson(sFile, facial_target_size, facial_classifier, face_db)

                    # Ask name or set scaned person
                    if person == False:
                        message('Unknown face - pleas identify')

                        # Get name of user
                        name = start_Listening(option='askName')
                        if name == None:
                            name = start_Listening(option='askName')
                        # Add user or stop asking
                        if name not in ['herken mij niet', 'tot ziens', 'niemand']:
                            status, person, face_db = DB.insertPerson(name, db, sFile)
                        else:
                            resetScanedPerson()

                    if person != False:
                        # Welcome and set scaned person
                        message('welcome %s'% person['first_name'])
                        UI.happy(async=True)
                        config.scaned_person = person

                        # Start welcoming person
                        welcome_thread = threading.Thread(name='welome Person', target=speech.welcomePerson, args=(person['first_name'],))
                        welcome_thread.start()

                        # Meanwile welcoming predict emotional state
                        emotion_recognition = AI.Emotion_Recognizer(emotion_classifier)
                        emotion_text = emotion_recognition.predict(frame)
                        screenUtil(emotion_text, verbose=config.VERBOSE)

                        # Wait until welcoming is done then interact with emotion
                        welcome_thread.join()
                        emotion_recognition.interact(speech, emotion_text)

                        # Restore Listener
                        listener.start()


            # if config.VERBOSE:
            #     out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            #     out = cv2.resize(out, (0, 0), fx=.5, fy=.5)
            #     cv2.imshow('window_frame', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    listener.pause()


def start_Listening(option=''):
    global speech, listener

    if config.active_mode:
        return
    else:
        config.active_mode = True

    if option == 'listen':
        listener.pause()
        speech.ping()
        message('Ik luister...')
        if speech.listen():
            config.active_mode = False
        listener.start()
        return

    if option == 'askName':
        listener.pause()
        speech.ping()
        message('Ik luister...')
        name = speech.getName()
        config.active_mode = False
        return name

    message("Hotkeys detected")
    speech.response()
    listener.pause()

    # Start Listening
    speech.ping()
    message('Ik luister...')
    speech.listen()

    # Start Listener again
    listener.start()
    config.active_mode = False
