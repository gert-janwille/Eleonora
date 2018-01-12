import cv2
import numpy as np
import eleonora.utils.config as config
from keras.models import load_model, model_from_json

from eleonora.modules import AI, DB
from eleonora.utils.util import *
from eleonora.utils.input import quit

def detected_callback():
    # TODO: SPEAK & LISTEN (person is available in config.scaned_person)
    #       - multiple keywords
    #       OR
    #       - stop snowboy and start speech_recognition (internet connection needed)
    #       - Let Eleonora speak gtts
    #
    # https://ggulati.wordpress.com/2016/02/24/coding-jarvis-in-python-3-in-2016/

    print ("hotword detected")

def main():
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
    listener = AI.HotKeyListener().listen(detected_callback)
    # Initialize SpeechRecognition
    speech = AI.SpeechRecognition(lang='nl')
    speech.talk('Hallo iedereen')


    # Read Database
    face_db = db.read(key="persons")

    cap = cv2.VideoCapture(0)
    samePerson = (False, 0)

    while config.process_this_frame:
        # Read cap and preprocess frame
        frame = cap.read()[1]
        frame, out = convertSequence(frame, grayscale=False)

        # Detect Faces, same person and output file
        flag, samePerson, frame, sFile = detect_faces(frame, emotion_target_size, samePerson[1])

        if config.VERBOSE:
            print(config.scaned_person)

        # Reset Person
        if not flag:
            resetScanedPerson()

        # Reset Time
        if flag and config.scaned_person:
            config.reset_time = 10

        # If no scaned person detect one
        if flag and not config.scaned_person:
            print('+++ face detected and scaned_person is empty')

            # Read Database
            face_db = db.read(key="persons")

            if not samePerson[0]:
                if db.isEmpty("persons"):
                    # add face to db
                    status, person, face_db = DB.insertPerson(db, sFile)
                else:
                    # Loop over db
                    person = identifyPerson(sFile, facial_target_size, facial_classifier, face_db)

                # Ask name or set scaned person
                if person == False:
                    print('ask name')
                    status, person, face_db = DB.insertPerson(db, sFile)
                else:
                    print('welcome %s'% person['first_name'])
                    config.scaned_person = person

            # # Predict Emotions
            # emotion_recognition = AI.Emotion_Recognizer(emotion_classifier)
            # emotion_text = emotion_recognition.predict(frame)
            # print(emotion_text)

            # screenUtil(emotion_text, verbose=config.VERBOSE)

        # if config.VERBOSE:
        #     out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        #     out = cv2.resize(out, (0, 0), fx=.5, fy=.5)
        #     cv2.imshow('window_frame', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    listener.stop()


if __name__ == '__main__':
    main()
