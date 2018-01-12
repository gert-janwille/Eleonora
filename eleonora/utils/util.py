import cv2
import dlib
import numpy as np
import eleonora.utils.config as config
from statistics import mode
from eleonora.modules import AI

face_detection = cv2.CascadeClassifier(config.detection_model_path)

def identifyPerson(frame, sizes, model, db):
    facialFrame = convertFacial(frame, (sizes))
    facial_recognition = AI.Facial_Recognizer(model, sizes)
    person = facial_recognition.verify(facialFrame, db=db, key='face')
    return person

def detect_faces(frame, sizes, prefaces):
    faces = face_detection.detectMultiScale(frame, 1.3, 5)
    samePerson = (False, len(faces)) if len(faces) != prefaces else (True, len(faces))

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, config.emotion_offsets)
        frame = sFile = frame[y1:y2, x1:x2]
        # cv2.imwrite('file.png', sFile)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (sizes))
            # cv2.imwrite('grey.png', frame)
        except:
            continue

        frame = preprocess_input(frame, True)
        frame = expand_dims(frame)

        return True, samePerson, frame, sFile
    return False, samePerson, frame, None

def getVerifyFile(path, sizes, prefix = None):
    f = cv2.imread(prefix + path)
    f = cv2.resize(f, sizes)
    f = np.expand_dims(f, 0)
    return f

def convertFacial(f, sizes):
    f = cv2.resize(f, sizes)
    f = np.expand_dims(f, 0)
    return f

def resetScanedPerson():
    config.reset_time = config.reset_time - 1
    if config.VERBOSE:
        print(config.reset_time)
    if config.reset_time <= 0:
        print('clear scaned person')
        config.scaned_person = []
        config.reset_time = 0

def getJson(path):
    f = open(path, 'r')
    json = f.read()
    f.close()
    return json

def readDatabase(path, key=False):
    data = json.loads(getJson(path))
    if key == False:
        return False, data

    db = data.get(key)
    isEmpty = False if len(data.get(key)) > 0 else True

    return isEmpty, db

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0, font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def convertSequence(f, sizes = (0.25, 0.25), grayscale=True):
    f = cv2.flip(f, 1)
    f = cv2.resize(f, (0, 0), fx=sizes[0], fy=sizes[1])
    if grayscale:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f

def crop_center(img, sizes=(20, 20)):
    cropx, cropy = sizes
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def expand_dims(f):
    f = np.expand_dims(f, 0)
    f = np.expand_dims(f, -1)
    return f

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def convertSequence(f, grayscale=True):
    f = cv2.flip(f, 1)
    out = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    if grayscale:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f, out

def getNames(n):
    n = n.split(' ')
    f, l = n[0], ' '.join(n[1:])
    return f, l

def startDetection(counter, arr, numDetections):
    if len(arr) > 0:
        if counter >= numDetections:
            return numDetections
        else:
            return counter + 1
    elif len(arr) == 0:
        return 0
    elif counter != 0:
        return counter - 1
    else:
        return 0

def screenUtil(txt, verbose=0):
    screens = config.screens
    screens.append(txt)

    if len(screens) > config.frame_window:
        screens.pop(0)
    try:
        emotion_mode = mode(screens)
    except:
        if verbose:
            print('error while setting emotion_mode')
